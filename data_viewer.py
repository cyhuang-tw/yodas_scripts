import gradio as gr
import argparse
import kaldiio
from pathlib import Path


def display_utterance(data_dir, identifier, search_by):
    """Display utterance text and audio."""

    data_dir = Path(data_dir)

    with open(data_dir / "text", 'r') as f_text, open(
        data_dir / "text.ctc", 'r') as f_ctc, open(
        data_dir / "text.prev", 'r') as f_prev, open(
        data_dir / "wav.scp", 'r') as f_wav:
        for i, (line_text, line_ctc, line_prev, line_wav) in enumerate(zip(f_text, f_ctc, f_prev, f_wav)):
            if search_by == "line_num" and i == int(identifier):
                break
            elif search_by == "utt_id" and line_text.startswith(identifier):
                break

    rate, speech = kaldiio.load_mat(line_wav.strip().split(maxsplit=1)[1])

    return line_text.strip().split(maxsplit=1)[1], line_ctc.strip().split(maxsplit=1)[1], line_prev.strip().split(maxsplit=1)[1], rate, (rate, speech)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to Kaldi data directory")
    args = parser.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("# Kaldi Utterance Viewer")
        with gr.Row():
            search_type = gr.Radio(
                choices=["line_num", "utt_id"],
                value="line_num",
                label="Search by"
            )
            identifier = gr.Textbox(label="Enter utterance ID or line number")

        btn = gr.Button("Display")
        text_output = gr.Textbox(label="text")
        text_ctc_output = gr.Textbox(label="text.ctc")
        text_prev_output = gr.Textbox(label="text.prev")
        rate_output = gr.Textbox(label="sampling rate")
        audio_output = gr.Audio(label="Audio")

        btn.click(
            fn=lambda x, y: display_utterance(args.data_dir, x, y),
            inputs=[identifier, search_type],
            outputs=[text_output, text_ctc_output, text_prev_output, rate_output, audio_output]
        )

    demo.launch(
        show_api=False,
        share=True,
        debug=True,
    )


if __name__ == "__main__":
    main()
