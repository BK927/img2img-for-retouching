import os
import sys
import traceback
import pandas as pd
from torch import randint

import modules.scripts as scripts
import gradio as gr

from modules import processing
from modules.processing import Processed, process_images, create_infotext
from PIL import Image, PngImagePlugin
from modules.shared import opts, state
from modules.script_callbacks import ImageSaveParams, before_image_saved_callback
from modules.sd_hijack import model_hijack

import importlib.util
import re

re_findidx = re.compile(
    r"(?=\S)(\d+)\.(?:[P|p][N|n][G|g]?|[J|j][P|p][G|g]?|[J|j][P|p][E|e][G|g]?|[W|w][E|e][B|b][P|p]?)\b"
)
re_findname = re.compile(r"[\w-]+?(?=\.)")


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def gr_show_value_none(visible=True):
    return {"value": None, "visible": visible, "__type__": "update"}


def gr_show_and_load(value=None, visible=True):
    if value:
        if value.orig_name.endswith(".csv"):
            value = pd.read_csv(value.name)
        else:
            value = pd.read_excel(value.name)
    else:
        visible = False
    return {"value": value, "visible": visible, "__type__": "update"}


class Script(scripts.Script):
    def title(self):
        return "Retouching images"

    def description(self):
        return "Process multiple images with loop and txt prompt"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return None

        with gr.Row():
            input_dir = gr.Textbox(label="Input directory", lines=1)
            output_dir = gr.Textbox(label="Output directory", lines=1)
        
        with gr.Row():
            is_traversing = gr.Checkbox(label="Traverse between the given denoising values.")

        with gr.Row():
            use_txt = gr.Checkbox(label="Use txt files as prompt.")

        with gr.Row(visible=False) as txt_dir_textbox:
            txt_dir = gr.Textbox(label="txt directory", lines=1)

        with gr.Row():
            is_rerun = gr.Checkbox(label="Loopback")

        with gr.Row(visible=False) as rerun_options:
            loops = gr.Slider(
                minimum=1,
                maximum=32,
                step=1,
                label="Loops",
                value=4,
                elem_id=self.elem_id("loops"),
            )
            denoising_strength_change_factor = gr.Slider(
                minimum=0.9,
                maximum=1.1,
                step=0.01,
                label="Denoising strength change factor",
                value=1,
                elem_id=self.elem_id("denoising_strength_change_factor"),
            )

        with gr.Row(visible=False) as traversing_options:
            traversing_start = gr.Slider(
                minimum=0.01,
                maximum=1,
                step=0.01,
                label="start",
                value=0.2,
            )
            traversing_end = gr.Slider(
                minimum=0.01,
                maximum=1,
                step=0.01,
                label="end",
                value=0.4,
            )
            traversing_step = gr.Slider(
                minimum=0.01,
                maximum=1,
                step=0.01,
                label="step",
                value=0.01,
            )

        is_rerun.change(
            fn=lambda x: gr_show(x),
            inputs=[is_rerun],
            outputs=[rerun_options],
        )
        use_txt.change(
            fn=lambda x: gr_show(x),
            inputs=[use_txt],
            outputs=[txt_dir_textbox],
        )
        is_traversing.change(
            fn=lambda x: gr_show(x),
            inputs=[is_traversing],
            outputs=[traversing_options],
        )

        return [
            input_dir,
            output_dir,
            use_txt,
            txt_dir,
            is_traversing,
            is_rerun,
            loops,
            denoising_strength_change_factor,
            traversing_start,
            traversing_end,
            traversing_step,
        ]

    def __create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def __add_denoising(self, float1, float2):
        int1 = int(float1 * 100)
        int2 = int(float2 * 100)
        return float(int1 + int2) / 100

    def run(
        self,
        p,
        input_dir,
        output_dir,
        use_txt,
        txt_dir,
        is_traversing,
        is_rerun,
        loops,
        denoising_strength_change_factor,
        traversing_start,
        traversing_end,
        traversing_step,
    ):

        initial_info = None
        
        images = [
            file
            for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
            if os.path.isfile(file)
        ]

        print(f'Will process following files: {", ".join(images)}')

        p.img_len = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        batch_count = p.n_iter

        p.batch_size = 1
        p.n_iter = 1

        init_prompt = p.prompt
        img_len = len(images)

        state.job_count = img_len * loops * batch_count
        
        if is_traversing:
            state.job_count = state.job_count * int((traversing_end - traversing_start) / traversing_step)


        txt_dict = {}
        if use_txt:
            for filename in os.listdir(txt_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(txt_dir, filename)
                    txt_dict[os.path.splitext(filename)[0]] = file_path

        if is_traversing:
            p.denoising_strength = traversing_start
            print(p.denoising_strength)

        for n in range(batch_count):
            processing.fix_seed(p)
            initial_output_dir = output_dir
            output_dir = os.path.join(output_dir, str(p.seed))
            base_output_dir = output_dir

            while True:
                current_denoising = p.denoising_strength
                # image iteration processs begins here
                for idx, path in enumerate(images):
                    if state.interrupted:
                        break
                    batch_images = []
                    print(f"Processing: {path}")
                    try:
                        img = Image.open(path)
                        batch_images.append((img, path))

                    except BaseException:
                        print(f"Error processing {path}:", file=sys.stderr)
                        print(traceback.format_exc(), file=sys.stderr)

                    if len(batch_images) == 0:
                        print("No images will be processed.")
                        break

                    state.job = f"{idx} out of {img_len}: {batch_images[0][1]}"
                    p.init_images = [x[0] for x in batch_images]

                    if use_txt:
                        img_name = os.path.splitext(os.path.basename(path))[0]
                        print("img name: " + img_name)
                        if img_name in txt_dict:
                            with open(txt_dict[img_name], "r") as file:
                                tags = file.readline()
                                print("tags: " + tags)
                                p.prompt = init_prompt + ", " + tags

                    proc = None
                    all_images = []

                    if is_rerun:
                        p.extra_generation_params = {
                            "Denoising strength change factor": denoising_strength_change_factor,
                        }


                        output_images, info = None, None
                        initial_seed = None
                        initial_info = None

                        initial_color_corrections = [
                            processing.setup_color_correction(p.init_images[0])
                        ]

                        history = []

                        for i in range(loops):
                            p.n_iter = 1
                            p.batch_size = 1
                            p.do_not_save_grid = True

                            if opts.img2img_color_correction:
                                p.color_corrections = initial_color_corrections

                            state.job = f"Iteration {i + 1}/{loops}"

                            proc = process_images(p)

                            if initial_seed is None:
                                initial_seed = proc.seed
                                initial_info = proc.info

                            init_img = proc.images[0]

                            p.init_images = [init_img]
                            p.seed = proc.seed + 1
                            p.denoising_strength = min(
                                max(
                                    p.denoising_strength * denoising_strength_change_factor, 0.1
                                ),
                                1,
                            )
                            history.append(proc.images[0])

                        proc.seed = initial_seed
                        p.denoising_strength = current_denoising
                        all_images += history
                    else:
                        proc = process_images(p)

                    if initial_info is None:
                        initial_info = proc.info

                    output_images = proc.images
                    if is_rerun:
                        output_images = all_images

                    for output, (input_img, path) in zip(proc.images, batch_images):
                        base_name = os.path.basename(path)

                        # 이미지 저장 함수
                        def save_image(img, filename, save_dir):
                            comments = {}
                            if len(model_hijack.comments) > 0:
                                for comment in model_hijack.comments:
                                    comments[comment] = 1

                            info = create_infotext(
                                p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, 0, 0
                            )
                            pnginfo = {}
                            if info is not None:
                                pnginfo["parameters"] = info

                            params = ImageSaveParams(img, p, filename, pnginfo)
                            before_image_saved_callback(params)
                            fullfn_without_extension, extension = os.path.splitext(filename)

                            info = params.pnginfo.get("parameters", None)

                            def exif_bytes():
                                return piexif.dump(
                                    {
                                        "Exif": {
                                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                                info or "", encoding="unicode"
                                            )
                                        },
                                    }
                                )

                            if extension.lower() == ".png":
                                pnginfo_data = PngImagePlugin.PngInfo()
                                for k, v in params.pnginfo.items():
                                    pnginfo_data.add_text(k, str(v))

                                img.save(
                                    os.path.join(save_dir, filename), pnginfo=pnginfo_data
                                )

                            elif extension.lower() in (".jpg", ".jpeg", ".webp"):
                                img.save(os.path.join(save_dir, filename))

                                if opts.enable_pnginfo and info is not None:
                                    piexif.insert(
                                        exif_bytes(), os.path.join(save_dir, filename)
                                    )
                            else:
                                img.save(os.path.join(save_dir, filename))
                        
                        # 실제 저장 하는 부분
                        if is_traversing:
                            output_dir = os.path.join(output_dir, str(round(p.denoising_strength, 2)))

                        if is_rerun:
                            for output_index, o in enumerate(all_images):
                                temp = output_dir
                                output_dir = os.path.join(
                                    output_dir, "loop" + str(output_index)
                                )
                                self.__create_folder(output_dir)
                                save_image(o, base_name, output_dir)
                                output_dir = temp
                        else:
                            self.__create_folder(output_dir)
                            save_image(output, base_name, output_dir)

                        output_dir = base_output_dir
                
                if is_traversing is False:
                    break
                else:
                    p.denoising_strength = self.__add_denoising(p.denoising_strength, traversing_step)
                    if not p.denoising_strength <= traversing_end:
                        p.denoising_strength = current_denoising
                        break

            output_dir = initial_output_dir
            if is_traversing:
                p.denoising_strength = traversing_start

        return Processed(p, [], p.seed, initial_info)
