from flask import Flask, request, jsonify
import argparse
import torch
from torchvision import transforms
import os
import re
from transformer_net import TransformerNet
import utils  # assuming utils module is present in your environment

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_model = TransformerNet()  # assuming TransformerNet class is defined in your environment

# 使用 os.environ.get 从环境变量中获取键为 'MODEL_PATH' 的值。如果没有找到，返回默认值 'saved_models/mosaic.pth'
model_path = os.environ.get('MODEL_PATH', 'saved_models/mosaic.pth')  
state_dict = torch.load(model_path)  # load your model

# remove saved deprecated running_* keys in InstanceNorm from the checkpoint
for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]

style_model.load_state_dict(state_dict)
style_model.to(device)
style_model.eval()

def stylize(content_image_path, output_image_path):
    content_scale = None
    content_image = utils.load_image(content_image_path, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = style_model(content_image).cpu()
    utils.save_image(output_image_path, output[0])

@app.route("/ns/process", methods=["POST"])
def stylize_api():
    content = request.json
    stylize(content['input_path'], content['output_path'])
    return jsonify({'output_path': content['output_path']})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
