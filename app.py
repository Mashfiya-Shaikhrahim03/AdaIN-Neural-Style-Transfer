
import os

import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired, NumberRange
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY", "dev-only-change-this-secret"
)
app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per request

app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

Bootstrap(app)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# --------------------------------------------------
# Upload form
# --------------------------------------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[
            InputRequired(),
            NumberRange(min=0.0, max=1.0),
        ],
    )

    submit = SubmitField("Transfer Style")


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_path = os.path.join(BASE_DIR, "vgg_normalised.pth")

decoder_path = r"D:\NST_teacher_copy\NST_Code\experiment\final_exp\decoder_final.pth"


if not os.path.isfile(vgg_path):
    raise FileNotFoundError(
        f"VGG weights not found: {vgg_path}"
    )

if not os.path.isfile(decoder_path):
    raise FileNotFoundError(
        f"Decoder weights not found: {decoder_path}"
    )

encoder = VGGEncoder(vgg_path).to(device)
decoder = Decoder().to(device)

decoder_state = torch.load(
    decoder_path,
    map_location=device,
    weights_only=True,
)
decoder.load_state_dict(decoder_state)

encoder.eval()
decoder.eval()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    content_tensor = (
        content_transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        style_transform(style_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(
            content_feats, style_feats
        )

        stylized_feats = (
            alpha * stylized_feats
            + (1.0 - alpha) * content_feats
        )

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.detach().cpu().clone()
    image = image.squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def save_upload(file_storage):
    """Validate and save an uploaded image with a unique filename."""
    if not file_storage or not file_storage.filename:
        return None

    if not allowed_file(file_storage.filename):
        raise ValueError("Please upload a PNG, JPG, or JPEG image.")

    original_name = secure_filename(file_storage.filename)

    if not original_name:
        raise ValueError("The uploaded filename is invalid.")

    # Verify that the uploaded file is actually a readable image.
    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as image:
            image.verify()
        file_storage.stream.seek(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError("The uploaded file is not a valid image.")

    # Avoid overwriting an existing upload with the same filename.
    filename = f"{os.urandom(8).hex()}_{original_name}"
    destination = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file_storage.save(destination)
    return filename


# --------------------------------------------------
# Main page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    # Only process uploads after the user submits the form.
    if form.validate_on_submit():
        try:
            content_filename = save_upload(form.content.data)
            style_filename = save_upload(form.style.data)

            if not content_filename or not style_filename:
                raise ValueError(
                    "Please upload both a content image and a style image."
                )

            content_path = os.path.join(
                app.config["UPLOAD_FOLDER"], content_filename
            )
            style_path = os.path.join(
                app.config["UPLOAD_FOLDER"], style_filename
            )

            with Image.open(content_path) as image:
                content_image = image.convert("RGB")

            with Image.open(style_path) as image:
                style_image = image.convert("RGB")

            alpha = float(form.alpha.data)

            stylized_image = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                alpha,
                device,
            )

            result_filename = f"stylized_{content_filename}"
            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"], result_filename
            )

            save_image(stylized_image, result_path)
            result_image = result_filename

        except (ValueError, OSError, UnidentifiedImageError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Style transfer failed.")
            error = (
                "Style transfer failed. Check the terminal for the "
                "technical error details."
            )

    elif form.is_submitted():
        error = "Please check the form fields and try again."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


# --------------------------------------------------
# Serve uploaded images and examples
# --------------------------------------------------

@app.route("/uploads/<path:filename>")
def send_image(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename
    )



@app.route('/examples/<path:filename>')
def send_example(filename):
    examples_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "examples"
    )
    return send_from_directory(examples_dir, filename)

    return send_from_directory(examples_dir, filename)


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    from werkzeug.serving import run_simple

    run_simple(
        "127.0.0.1",
        5000,
        app,
        use_reloader=False,
        use_debugger=False,
    )
