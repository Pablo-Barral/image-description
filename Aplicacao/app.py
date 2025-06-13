import os
from datetime import datetime

from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carregar os modelos do BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Função para gerar descrição
def gerar_descricao(imagem_path):
    raw_image = Image.open(imagem_path).convert("RGB")  # Usando Pillow para abrir a imagem
    inputs = processor(raw_image, return_tensors="pt")  # Processando a imagem para o modelo BLIP
    out = model.generate(**inputs)  # Gerando a descrição com o modelo BLIP
    descricao = processor.decode(out[0], skip_special_tokens=True)  # Decodificando a descrição gerada

    # Traduz descrição completa para português
    descricao_portugues = GoogleTranslator(source='en', target='pt').translate(descricao)
    return descricao_portugues


# Função para converter texto em áudio
def texto_para_audio(descricao):
    tts = gTTS(descricao, lang='pt')
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "descricao.mp3")
    tts.save(audio_path)
    return audio_path


# Rota principal
@app.route('/')
def home():
    return render_template('index.html')

# Rota para processar o upload da imagem
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'imagem' not in request.files:
        return 'Nenhuma imagem fornecida', 400

    file = request.files['imagem']
    if file.filename == '':
        return 'Nenhuma imagem selecionada', 400

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'imagem.jpg')
        file.save(filename)

        # Gerar a descrição e o áudio
        descricao = gerar_descricao(filename)
        audio_path = texto_para_audio(descricao)

        # Usar timestamp para forçar atualização no navegador
        timestamp = datetime.now().timestamp()


        # Retornar a descrição e o link para o áudio
        return render_template('index.html', descricao=descricao, audio_path=audio_path, timestamp=timestamp)


# Rota para servir o arquivo de áudio gerado
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
