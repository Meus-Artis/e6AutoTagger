import json
import os
import string
from io import BytesIO
import gradio as gr
import pandas as pd
from PIL import Image
import safetensors.torch
import timm
from timm.models import VisionTransformer
import torch
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from math import ceil
from typing import Callable, List
from functools import partial
from huggingface_hub import hf_hub_download # download model from hugging face
import os.path
torch.set_grad_enabled(False)

print("Checking for model...") # Download dataset
if not os.path.isfile("./JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors"):
    print("Downloading tagger model, please wait...")
    hf_hub_download(repo_id="RedRocket/JointTaggerProject", filename="JTP_PILOT2/JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors", repo_type="model", local_dir="../")
else:
    print("JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors found")

if not os.path.isfile("./tags.json"):
    print("Downloading tags.json")
    hf_hub_download(repo_id="RedRocket/JointTaggerProject", filename="JTP_PILOT2/tags.json", repo_type="model", local_dir="../")
else:
    print("tags.json found")

print("Model check complete!")

class Fit(torch.nn.Module):
    def __init__(
            self,
            bounds: tuple[int, int] | int,
            interpolation=InterpolationMode.LANCZOS,
            grow: bool = True,
            pad: float | None = None
    ):
        super().__init__()
        self.bounds = (bounds, bounds) if isinstance(bounds, int) else bounds
        self.interpolation = interpolation
        self.grow = grow
        self.pad = pad

    def forward(self, img: Image) -> Image:
        wimg, himg = img.size
        hbound, wbound = self.bounds
        hscale = hbound / himg
        wscale = wbound / wimg
        if not self.grow:
            hscale = min(hscale, 1.0)
            wscale = min(wscale, 1.0)
        scale = min(hscale, wscale)
        if scale == 1.0:
            return img
        hnew = min(round(himg * scale), hbound)
        wnew = min(round(wimg * scale), wbound)
        img = TF.resize(img, (hnew, wnew), self.interpolation)
        if self.pad is None:
            return img
        hpad = hbound - hnew
        wpad = wbound - wnew
        tpad = hpad // 2
        bpad = hpad - tpad
        lpad = wpad // 2
        rpad = wpad - lpad
        return TF.pad(img, (lpad, tpad, rpad, bpad), self.pad)

    def __repr__(self) -> str:
        return (
                f"{self.__class__.__name__}(" +
                f"bounds={self.bounds}, " +
                f"interpolation={self.interpolation.value}, " +
                f"grow={self.grow}, " +
                f"pad={self.pad})"
        )

class CompositeAlpha(torch.nn.Module):
    def __init__(
            self,
            background: tuple[float, float, float] | float,
    ):
        super().__init__()
        self.background = (background, background, background) if isinstance(background, float) else background
        self.background = torch.tensor(self.background).unsqueeze(1).unsqueeze(2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-3] == 3:
            return img
        alpha = img[..., 3, None, :, :]
        img[..., :3, :, :] *= alpha
        background = self.background.expand(-1, img.shape[-2], img.shape[-1])
        if background.ndim == 1:
            background = background[:, None, None]
        elif background.ndim == 2:
            background = background[None, :, :]
        img[..., :3, :, :] += (1.0 - alpha) * background
        return img[..., :3, :, :]

    def __repr__(self) -> str:
        return (
                f"{self.__class__.__name__}(" +
                f"background={self.background})"
        )

transform = transforms.Compose([
    Fit((384, 384)),
    transforms.ToTensor(),
    CompositeAlpha(0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    transforms.CenterCrop((384, 384)),
])

model = timm.create_model(
    "vit_so400m_patch14_siglip_384.webli",
    pretrained=False,
    num_classes=9083,
)

class GatedHead(torch.nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes * 2)
        self.act = torch.nn.Sigmoid()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])
        return x

model.head = GatedHead(min(model.head.weight.shape), 9083)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device(device_name)
offload_device = torch.device("cpu")
if device_dtype == torch.float16:
    print("Running on " + device_name + " with F16 Precision")
else:
    print("Running on " + device_name + " with F32 Precision")
try:
    safetensors.torch.load_model(model, "./JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors")
    model.eval()
except FileNotFoundError or IOError:
    print('You might be missing JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors file! Please download the JTP PILOT 2 safetensors (and tags.json) from https://huggingface.co/RedRocket/JointTaggerProject')
    exit(-1)

try:
    with open("./tags.json", "r") as file:
        tags = json.load(file)
        #print(tags)
    allowed_tags = list(tags.keys())
    #print(allowed_tags)
except FileNotFoundError or IOError:
    print('You might be missing tags.json file! Please download and place it in the folder. https://huggingface.co/RedRocket/JointTaggerProject')
    exit(-1)

for idx, tag in enumerate(allowed_tags):
    allowed_tags[idx] = tag.replace("_", "_")
sorted_tag_score = {}

def run_classifier(image, threshold):
    model.to(device=device, dtype=device_dtype)
    global sorted_tag_score
    img = image.convert('RGBA')
    tensor = transform(img).unsqueeze(0).to(device, dtype=device_dtype)
    with torch.no_grad():
        probits = model(tensor)[0]
        values, indices = probits.topk(250)
    tag_score = dict()
    for i in range(indices.size(0)):
        tag_score[allowed_tags[indices[i]]] = values[i].item()
    sorted_tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))
    model.to(device=offload_device, dtype=device_dtype)
    return create_tags(threshold)

def create_tags(threshold):
    global sorted_tag_score
    filtered_tag_score = {key: value for key, value in sorted_tag_score.items() if value > threshold}
    text_no_impl = " ".join(filtered_tag_score.keys())
    return text_no_impl, filtered_tag_score

def clear_image():
    global sorted_tag_score
    sorted_tag_score = {}
    return "", {}

class ImageDataset(Dataset):
    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform
        self.image_path = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), os.path.basename(img_path)

def measure_duration(images, threshold) -> int:
    return ceil(len(images) / 64) * 5 + 3

def process_images(images, threshold, prepend_tags, blacklist_tags):
    dataset = ImageDataset(images, transform) # list of file paths
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True, drop_last=False)
    all_results: list[tuple[string, string, string]] = []
    with torch.no_grad():
        model.to(device, dtype=device_dtype)
        for batch, filenames, in dataloader:
            batch = batch.to(device)
            probabilities = model(batch)
            for i, prob in enumerate(probabilities):
                indices = torch.where(prob > threshold)[0]
                values = prob[indices]
                temp = []
                tag_score = dict()
                for j in range(indices.size(0)):
                    tag = allowed_tags[indices[j]]
                    if tag in blacklist_tags:
                        continue
                    score = values[j].item()
                    temp.append([tag, score])
                    tag_score[tag] = score
                tags = ", ".join([prepend_tags] + [t[0] for t in temp])
                all_results.append((filenames[i], tags, tag_score))
    model.to(offload_device, dtype=torch.float32)
    return all_results

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def process_folder(folder_input, threshold, prepend_tags, blacklist_tags): # local folder
    if folder_input is None:
        return None
    all_files = [os.path.join(folder_input, f) for f in os.listdir(folder_input) if os.path.isfile(os.path.join(folder_input, f))]
    image_files = [f for f in all_files if is_valid_image(f)]
    results = process_images(image_files, threshold, prepend_tags, blacklist_tags)
    for file, text_no_impl, _ in results:
        txt_filename_with_path = os.path.join(folder_input, f"{os.path.splitext(file)[0]}.txt")
        with open(txt_filename_with_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_no_impl)
    df = pd.DataFrame([(os.path.basename(f), t) for f, t, _ in results], columns=['Image', 'Tags'])
    return df

with gr.Blocks(css=".output-class { display: none; }") as ui:
    gr.Markdown("""## Joint Tagger Project: JTP-PILOT²
    This tagger is designed for use on furry images (though may very well work on out-of-distribution images, potentially with funny results).  A threshold of 0.2 is recommended.  Lower thresholds often turn up more valid tags, but can also result in some amount of hallucinated tags.

    This tagger is the result of joint efforts between members of the RedRocket team, with distinctions given to Thessalo for creating the foundation for this project with his efforts, RedHotTensors for redesigning the process into a second-order method that models information expectation, and drhead for dataset prep, creation of training code and supervision of training runs.
    Special thanks to Minotoro at frosting.ai for providing the compute power for this project.

    Additional changes (namely, to make this UI easier to self-host) have been made by the likes of Reclusiarch, Tylor, and Velvet on Discord, with Meus Artis making further additional changes to adapt this for use with e621.""")
    with gr.Tabs():
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Source", type='pil', height=512, show_label=False)
                    threshold_slider = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.20, label="Threshold")
                with gr.Column():
                    tag_string = gr.Textbox(label="Tag String")
                    label_box = gr.Label(label="Tag Predictions", num_top_classes=250, show_label=False)
            image_input.upload(
                fn=run_classifier,
                inputs=[image_input, threshold_slider],
                outputs=[tag_string, label_box]
            )
            threshold_slider.input(
                fn=create_tags,
                inputs=[threshold_slider],
                outputs=[tag_string, label_box]
            )
        with gr.TabItem("Local Folder"):
            gr.Markdown("""This tab processes an entire folder at once.
            # ⚠ Processing a folder will overwrite any existing captions! ⚠""")
            with gr.Row():
                with gr.Column():
                    prepend_tags = gr.Textbox(label="Prepend")
                    blacklist_tags = gr.Textbox(label="Blacklist Tags")
                    #folder_input = gr.Files(label="Select Local Folder", file_count="directory")
                    folder_input = gr.Textbox(label="Path to dataset folder") # Gradio kinda sucks for this - there's no 'select a folder' component like in other UI libs :(
                    multi_threshold_slider = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.20, label="Threshold")
                    process_button = gr.Button("Process Folder")
                with gr.Column():
                    #zip_output = gr.File(label="Download Tagged Text Files (ZIP)")
                    dataframe_output = gr.Dataframe(label="Image Tags Summary") # TODO: Make the slider update tags in real time?
            process_button.click(
                fn=process_folder,
                inputs=[folder_input, multi_threshold_slider, prepend_tags, blacklist_tags],
                outputs=[dataframe_output]
            )

if __name__ == "__main__":
    ui.launch(server_name="127.0.0.1",server_port=22870)