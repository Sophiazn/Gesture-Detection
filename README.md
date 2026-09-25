# Gesture-Detection
Gesture Detection on PC

Versão adaptada do projeto Gesture Detection on Arduino UNO Q using Edge Impulse para rodar 100% localmente no seu PC, sem placa Arduino, sem Arduino App Lab, sem Flasher CLI e sem conta no Edge Impulse.

O que muda em relação ao projeto original
Original (Arduino UNO Q)	Esta versão (PC)
Placa Arduino UNO Q + hub USB-C + fonte 5V/3A	Só o seu PC
Arduino App Lab / Bricks / app.yaml	Script Python simples
Modelo treinado e exportado do Edge Impulse	Modelo pré-treinado do MediaPipe (baixado automaticamente)
Interface web (arduino:web_ui)	Janela OpenCV local
Requisitos
Python 3.9+
Webcam USB comum
Conexão com a internet na primeira execução (só para baixar o modelo, ~8 MB)
Instalação
bash
pip install -r requirements.txt
Execução
bash
python main.py

Uma janela abrirá mostrando o vídeo da webcam com:

caixa delimitadora ao redor da mão detectada
rótulo do gesto reconhecido e a confiança do modelo
pontos dos landmarks da mão

Pressione q para encerrar.

Gestos reconhecidos

O modelo pré-treinado do MediaPipe reconhece 7 gestos fixos: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.

Importante: esse conjunto é genérico (não é o alfabeto de Libras). Para gestos customizados — como os sinais do seu projeto de Libras — não é possível usar este modelo pronto; seria necessário treinar um classificador próprio (o que você já está fazendo com MediaPipe + Random Forest) ou treinar um modelo customizado no Edge Impulse e exportá-lo para o alvo "Linux (x86)".

Alternativa: usar o mesmo modelo do Edge Impulse (se você tiver um treinado)

Caso já tenha um modelo próprio treinado no Edge Impulse (por exemplo, treinado para os sinais de Libras), dá para rodá-lo neste mesmo PC, sem Arduino:

No Edge Impulse Studio, na etapa de Deployment, escolha o alvo "Linux (x86)" em vez de "Arduino UNO Q" e baixe o .eim.
Instale o runner:
bash
   npm install -g edge-impulse-linux
Rode:
bash
   edge-impulse-linux-runner --model-file caminho/para/seu-modelo.eim

Isso abre a webcam local e faz a inferência com o SEU modelo, sem placa nenhuma.
