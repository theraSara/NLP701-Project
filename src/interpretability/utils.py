import os
import json
import html

def load_config(config_path):
    # load configuration from a JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def visualize_attributions_html(tokens, attributions, pred_label, pred_score, true_label=None, sample_id=None):
    # Create an HTML string highlighting tokens with colors
    # the colors are based on attribution scroes, with intensity proportional to magnitude.
    # Positive attributions are green
    # Negative attributions are red
    max_attr = max(abs(min(attributions)), abs(max(attributions))) if attributions else 0.0
    highlighted_text = ""
    for token, score in zip(tokens, attributions):
        # normalize intensity
        frac = abs(score) / max_attr if max_attr > 1e-9 else 0.0
        
        # background color (green (positive), red (negative), white (zero))
        if score > 0:
            # green
            r = int(255 * (1-frac))
            g = 255
            b = int(255 * (1-frac))
        elif score < 0:
            # red
            r = 255
            g = int(255 * (1-frac))
            b = int(255 * (1-frac))
        else:
            r = 255
            g = 255
            b = 255
        color = f"rgb({r},{g},{b})"
        token_esc = html.escape(token)
        highlighted_text += f'<span style="background-color: {color}; padding: 2px; border-radius:2px;">{token_esc}</span> '

    # header info with prediction and true label
    header_parts = []
    if sample_id is not None:
        header_parts.append(f"Sample {sample_id}")
    if pred_label is not None:
        header_parts.append(f"Prediction: {pred_label} (score {pred_score:.2f})")
    if true_label is not None:
        header_parts.append(f"True label: {true_label}")
    header_html = f"<p><b>{' — '.join(header_parts)}</b></p>\n" if header_parts else ""
    body_html = f"<p>{highlighted_text.strip()}<p>\n"
    return header_html + body_html

