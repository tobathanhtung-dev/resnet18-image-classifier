
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd

from torchvision import models, transforms
from PIL import Image

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Classificador de Imagens",
    page_icon="🏞️",
    layout="centered"
)

CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

MODEL_PATH = "resnet_best_model.pth"

# --------------------------------------------------
# 2. Load the trained ResNet-18 model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, 6)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


model = load_model()

# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 4. Application interface
# --------------------------------------------------

st.title("🏞️ Classificador de Imagens")

st.write(
    "Envie uma imagem para classificá-la utilizando "
    "uma rede neural ResNet-18."
)

uploaded_file = st.file_uploader(
    "Selecione uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Imagem enviada",
        width="stretch"
    )

    # Prepare image for prediction
    image_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Sort predictions
    results = sorted(
        [
            (CLASSES[i], probabilities[i].item() * 100)
            for i in range(len(CLASSES))
        ],
        key=lambda x: x[1],
        reverse=True
    )

    predicted_class, confidence = results[0]

    st.divider()

    st.subheader("Resultado da classificação")

    st.success(
        f"Categoria prevista: {predicted_class.upper()}"
    )

    st.metric(
        label="Pontuação da classe prevista",
        value=f"{confidence:.2f}%"
    )

    # Ambiguity warning
    difference = results[0][1] - results[1][1]

    if difference < 10:
        st.warning(
            "Classificação potencialmente ambígua: "
            "as duas categorias principais apresentam "
            "pontuações semelhantes."
        )

    st.divider()

    st.subheader("Pontuações por categoria")

    for class_name, score in results:

        st.write(f"**{class_name.capitalize()} — {score:.2f}%**")
        st.progress(score / 100)

    st.caption(
        "As pontuações representam valores obtidos pelo "
        "softmax e não são probabilidades calibradas."
    )
