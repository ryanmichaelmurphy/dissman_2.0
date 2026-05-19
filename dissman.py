from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.image import Image
from kivy.uix.vkeyboard import VKeyboard
from kivy.core.image import Image as CoreImage
from kivy.lang import Builder
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.clock import mainthread
import random
import base64
import io
import threading
from threading import Timer
from threading import Thread
import requests
import time
import os
import subprocess
import cv2
import numpy as np
from gpiozero import Button as GPIOButton
from gpiozero.exc import BadPinFactory
#from signal import pause
from escpos.printer import Usb
from openai import OpenAI
from pathlib import Path
import pyttsx3
from queue import Queue, Empty
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

GPIO_PIN = 17
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
coin_acceptor = None

# path to files
BASE_DIR = Path(__file__).resolve().parent
Builder.load_file(str(BASE_DIR / "insultmaster3.kv"))
path = str(BASE_DIR) + "/"

try:
    coin_acceptor = GPIOButton(GPIO_PIN)
except BadPinFactory:
    print("GPIO not available on this device; coin input disabled.")
except Exception as e:
    print(f"GPIO init failed ({e}); coin input disabled.")

try:
    p = Usb(0x0416, 0x5011, in_ep=0x81, out_ep=0x01, profile='POS-5890')
except Exception as e:
    print(f"Printer init failed ({e}); printing disabled.")
    p = None


_tts_q = Queue()
_tts_engine = None
_tts_worker_started = False

def _tts_worker():
    global _tts_engine
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    _tts_engine.setProperty("volume", 1.0)

    while True:
        try:
            text = _tts_q.get()
            if text is None:
                break
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

def speak(text: str):
    text = str(text)

    if sys.platform.startswith("win"):
        # Windows: use built-in System.Speech (reliable, repeatable)
        safe = str(text).replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak('{safe}');"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    elif sys.platform == "darwin":
        # macOS: use built-in 'say' command
        subprocess.Popen(
            ["say", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # Pi/Linux: espeak-ng (fast, offline)
        subprocess.Popen(
            ["espeak-ng", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class ImageJob:
    """Holds state for the in-flight or completed image generation."""

    def __init__(self):
        self.image_path = None
        self.ready = False
        self.error = False

    def reset(self):
        self.image_path = None
        self.ready = False
        self.error = False


def start_image_generation(source_image_path, job, out_path):
    """Fire the GPT image edit in a background thread. Updates `job` in place."""
    job.reset()

    def _work():
        try:
            with open(source_image_path, "rb") as f:
                response = client.images.edit(
                    model="gpt-image-1",
                    image=f,
                    prompt=(
                        "You are a middle school bully. Draw this person as a crude "
                        "middle school notebook doodle. Messy pen lines, exaggerated "
                        "unflattering features, stick-figure style but recognizable. "
                        "Make them uglier than they actually are with a stupid facial "
                        "expression."
                    ),
                    n=1,
                    size="1024x1024",
                )
            data = base64.b64decode(response.data[0].b64_json)
            with open(out_path, "wb") as f:
                f.write(data)
            job.image_path = out_path
            job.ready = True
        except Exception as e:
            print(f"[image-gen] failed: {e}")
            job.error = True

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '480')
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'fullscreen', '1')

from insult_store import InsultStore, CATEGORY_LABELS

INSULT_STORE = InsultStore(BASE_DIR / "insults")

# Category codes used internally: 'g', 'r', 'old', 'all'.
# Display labels come from CATEGORY_LABELS; 'all' is "Anything Goes".
CATEGORY_DISPLAY = [
    ("g", "G-rated"),
    ("r", "R-rated"),
    ("old", "Old-timey"),
    ("all", "Anything Goes"),
]

class ThemedButton(Button):
    def __init__(self, **kwargs):
        super(ThemedButton, self).__init__(**kwargs)
        self.font_name = App.get_running_app().fonts['button']
        self.font_size = '18sp'
        self.color = App.get_running_app().theme_colors['primary']
        self.background_normal = ''
        self.background_color = [0,0,0] #App.get_running_app().theme_colors['secondary']

        with self.canvas.before:
            self.border_color = Color(*App.get_running_app().theme_colors['primary'])  # Border color
            self.border_line = Line(rectangle=(self.x, self.y, self.width, self.height), width=2)  # Border width

        # Bind to size and position changes to redraw the border
        self.bind(pos=self.update_border, size=self.update_border)

    def update_border(self, *args):
        self.border_line.rectangle = (self.x, self.y, self.width, self.height)

class InsultScreen(Screen):
    def generate_insults(self, category: str) -> list[str]:
        adj_list = INSULT_STORE.adjectives(category)
        noun_list = INSULT_STORE.nouns(category)
        if not adj_list or not noun_list:
            return []
        return [
            f"{random.choice(adj_list)} {random.choice(noun_list)}"
            for _ in range(3)
        ]

    def on_enter(self, *args):
        self.ids.insult_options.clear_widgets()
        category = self.manager.current_category
        insults = self.generate_insults(category)
        for insult in insults:
            btn = ThemedButton(text=insult, size_hint_y=None, height=40)
            btn.bind(on_release=self.show_insult)
            self.ids.insult_options.add_widget(btn)
        self.ids.header.text = "What best describes you?"
        speak("Which insult best describes you?")

    def show_insult(self, instance):
        self.manager.get_screen('display').ids.insult_label.text = f"you {instance.text}."
        self.manager.transition.direction = 'left'
        app = App.get_running_app()
        if app.image_job.ready:
            self.manager.get_screen('display').ids.dall_e_image.source = app.image_job.image_path
            self.manager.current = 'display'
        else:
            self.manager.current = 'load'

class CameraScreen(Screen):
    def on_enter(self):
        speak("Let me get a good look at you")
        # Clear any previous image or terminal display
        os.system('clear')  # This clears the terminal screen
        self.ids.captured_image.source = ''
        self.setup_camera()

        if not hasattr(self, 'img1'):
            self.img1 = Image()
            self.add_widget(self.img1)

            self.overlay_text = Label(
                text="Let me get a good look at you",
                size_hint=(None, None),
                size=(400, 50),
                pos_hint={'center_x': 0.5, 'top': 1},
                font_name='FreeMono',
                font_size='24sp',
                color=[0, 0, 0]
            )
            self.add_widget(self.overlay_text)

        # Start webcam preview
        Clock.schedule_interval(self.update_preview, 1 / 30)  # ~30fps for smoother preview
        Clock.schedule_once(self.capture_image, 7)

    def setup_camera(self):
        # Initialize the camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            return

    def update_preview(self, dt):
        ret, frame = self.camera.read()
        if ret:
            # Display frame in a GUI element
            self.display_frame(frame)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img1.texture = texture

    def capture_image(self, dt):
        ret, frame = self.camera.read()
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=50)
        if not ret:
            return

        timestamp = str(int(time.time()))
        save_path = path + 'test_' + timestamp + '.png'
        cv2.imwrite(save_path, frame)
        App.get_running_app().last_image_path = save_path
        self.ids.captured_image.source = save_path

        Clock.unschedule(self.update_preview)
        if self.camera.isOpened():
            self.camera.release()

        app = App.get_running_app()
        out_path = f'{path}downloaded_image_{timestamp}.png'
        start_image_generation(save_path, app.image_job, out_path)

        Clock.schedule_once(self.go_to_insult, 1.5)

    def go_to_insult(self, dt):
        self.manager.transition.direction = 'left'
        self.manager.current = 'insult'

class LoadScreen(Screen):
    def on_enter(self, *args):
        if not hasattr(self, "image_widget"):
            self.image_widget = Image(source=f'{path}thinking0.png')
            self.add_widget(self.image_widget)
        self.current_image = 1
        speak("Thinking bad thoughts about you.")
        Clock.schedule_interval(self.check_image_ready, 0.3)

    def check_image_ready(self, dt):
        app = App.get_running_app()
        if app.image_job.error:
            self.manager.current = 'splash'
            return False

        num_images = 16
        self.current_image = (self.current_image + 1) % num_images
        self.image_widget.source = f'{path}thinking{self.current_image}.png'

        if app.image_job.ready:
            self.manager.get_screen('display').ids.dall_e_image.source = app.image_job.image_path
            self.manager.current = 'display'
            return False

class DisplayScreen(Screen):
    has_entered = False

    def on_enter(self, *args):
        if getattr(self, "has_entered", False):
            return
        self.has_entered = True

        # Add functionality to print image and insult text
        self.ids.qr_button.clear_widgets()  # Clear existing buttons
        self.ids.teach_button.clear_widgets()
        teach_btn = ThemedButton(
            text="Insult Dissman to teach him new insults",
            size_hint_y=None, height=40,
        )
        teach_btn.bind(on_release=lambda x: self.go_to_teach())
        self.ids.teach_button.add_widget(teach_btn)
        dall_e_image_path = self.ids.dall_e_image.source
        insult_text = self.ids.insult_label.text
        # insult_text format is "you [adj] [noun]."
        parts = insult_text.split(' ')

        # Schedule the speech with explicit delays for dramatic effect
        speak("This is what you look like.")

        # "you" after 2 seconds
        Clock.schedule_once(lambda dt: speak(parts[0]), 2.0)

        # [adj] after 3.2 seconds
        if len(parts) > 1:
            Clock.schedule_once(lambda dt: speak(parts[1]), 3.2)

        # [noun] after 4.7 seconds
        if len(parts) > 2:
            Clock.schedule_once(lambda dt: speak(parts[2]), 4.7)

        self.print_image_and_text(dall_e_image_path, insult_text, p)

        self.ids.qr_button.size_hint = (1, None)
        self.ids.qr_button.height = 40
        qr_button = ThemedButton(text='See the code and "artist" statement', size_hint_y=None, height=40)
        qr_button.bind(on_release=lambda x: self.print_qr(p))
        self.ids.qr_button.add_widget(qr_button)

        # Schedule transition to the splash screen after 10 seconds
        self._return_event = Clock.schedule_once(lambda x: self.cleanup_and_restart(), 10)

    def cleanup_and_restart(self):
        self.has_entered = False
        self.p = None  # Release the printer by removing the reference
        self.manager.current = 'splash'

    def go_to_teach(self):
        # Cancel the auto-return-to-splash timer; teach flow owns the return.
        if getattr(self, "_return_event", None):
            self._return_event.cancel()
        self.has_entered = False
        self.manager.transition.direction = 'left'
        self.manager.current = 'teach_category'

    def print_qr(self, p):
        # --- debounce so we don't print twice on a double-trigger ---
        now = time.time()
        # _last_qr_ts will default to 0 if it doesn't exist yet
        last = getattr(self, "_last_qr_ts", 0)
        if now - last < 2.0:
            print("DEBUG: print_qr debounced, ignoring extra tap")
            return
        self._last_qr_ts = now

        print("DEBUG: print_qr called")

        if p is None:
            print("DEBUG: print_qr called but printer is unavailable; skipping.")
            return

        p._raw(b'\x1b\x40')
        p.text("\n\n")
        p.text('link to code and\n')
        p.text(r'"artist" statement')
        p.text("\n\n")
        p.image(path+"qrcode_scaled.png")
        p.text("\n\n\n\n")
        p._raw(b'\x1b\x40')

    def print_image_and_text(self, image_path, text, p):
        from PIL import Image, ImageEnhance

        # Load and process the image
        image = Image.open(image_path)
        max_width = 380
        max_height = 380
        image = image.resize((max_width, max_height), Image.Resampling.LANCZOS)

        # Decrease the contrast of the image
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.3)

        # Save the processed image to a temporary path
        timestamp = str(int(time.time()))
        temp_image_path = path + "downloaded_image_"+ timestamp +".jpg"
        image.save(temp_image_path)

        try:
            # Clear the buffer before printing
            p._raw(b'\x1b\x40')  # ESC @ Initialize
            p.text("\n\n\n")
            p.text("This is what you look like...\n")
            p.text("\n")

            p.image(temp_image_path)

            p.text("\n\n")
            p.text(text)
            p.text("\n\n\n\n")
            p._raw(b'\x1b\x40')

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            for filename in os.listdir(path):
                if 'test_' in filename or 'downloaded_image_' in filename:
                    file_path = os.path.join(path, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

class TeachWordScreen(Screen):
    prompt = ""
    next_screen = ""
    pos_key = ""

    def on_enter(self, *args):
        self.ids.prompt_label.text = self.prompt
        self.ids.word_input.text = ""
        speak(self.prompt)

        self.ids.keyboard_holder.clear_widgets()
        kb = VKeyboard(layout='qwerty', size_hint=(1, 1))
        kb.bind(on_key_up=self._on_key)
        self.ids.keyboard_holder.add_widget(kb)

        self.ids.action_row.clear_widgets()
        cancel = ThemedButton(text="Cancel")
        cancel.bind(on_release=lambda x: self.cancel())
        submit = ThemedButton(text="Submit")
        submit.bind(on_release=lambda x: self.submit())
        self.ids.action_row.add_widget(cancel)
        self.ids.action_row.add_widget(submit)

    def _on_key(self, keyboard, key, *args):
        display, key_code, special, ascii_code = key
        current = self.ids.word_input.text
        if special == 'backspace':
            self.ids.word_input.text = current[:-1]
        elif special == 'enter':
            self.submit()
        elif special == 'spacebar':
            self.ids.word_input.text = current + ' '
        elif display and len(display) == 1:
            self.ids.word_input.text = current + display

    def submit(self):
        word = self.ids.word_input.text.strip().lower()
        if not word:
            speak("Type something first.")
            return
        app = App.get_running_app()
        app.teach_submission[self.pos_key] = word
        self.manager.transition.direction = 'left'
        self.manager.current = self.next_screen

    def cancel(self):
        App.get_running_app().teach_submission = {}
        self.manager.transition.direction = 'right'
        self.manager.current = 'splash'


class TeachAdjScreen(TeachWordScreen):
    prompt = "Type the adjective"
    pos_key = "adj"
    next_screen = "teach_noun"


class TeachNounScreen(TeachWordScreen):
    prompt = "Type the noun"
    pos_key = "noun"
    next_screen = "teach_submit"


class TeachCategoryScreen(Screen):
    def on_enter(self, *args):
        self.ids.teach_categories.clear_widgets()
        speak("Pick a category for your insult.")
        for code, label in [("g", "G-rated"), ("r", "R-rated"), ("old", "Old-timey")]:
            btn = ThemedButton(text=label, size_hint_y=None, height=50)
            btn.bind(on_release=lambda inst, c=code: self.select(c))
            self.ids.teach_categories.add_widget(btn)

    def select(self, category_code):
        App.get_running_app().teach_submission = {"category": category_code}
        self.manager.transition.direction = 'left'
        self.manager.current = 'teach_adj'

class CategoryScreen(Screen):
    def select_category(self, category_code: str):
        self.manager.current_category = category_code
        self.manager.transition.direction = 'left'
        self.manager.current = 'camera'

class SplashScreen(Screen):
    def on_enter(self, *args):
        # Cancel existing events if any to prevent stacking
        if hasattr(self, 'animation_event'):
            self.animation_event.cancel()
        if hasattr(self, 'insult_event'):
            self.insult_event.cancel()

        self.image_widget = Image(source=f'{path}insertcoin0.png')
        self.add_widget(self.image_widget)
        self.current_image = 1
        print("waiting for coin...")
        speak("Insert coin for insult.")

        self.insult_list = [
            "You suck",
            "Give me a quarter dork",
            "I can smell your balls from here",
            "You're a disgrace",
            "Who let you in?",
            # "What ya looking at?",
            "You call that a haircut?",
            # "Who let a clown in?",
            # "Did you get dressed in the dark?",
            "That look isn't working",
            "You're trying way too hard",
            # "Ever heard of confidence?",
            # "Yikes, rough day?",
            # "That's… unfortunate",
            "You call that style?",
            "You look like a before picture",
            # "You look like you lost a fight with a lawnmower",
            # "You're the reason God created the middle finger",
            # "You're as useless as a screen door on a submarine",
            "You're like a software update: whenever I see you, I think, 'Not now.'"
            ""

        ]
        self.current_insults = list(self.insult_list)
        random.shuffle(self.current_insults)
        self.last_insult = None

        self.animation_event = Clock.schedule_interval(self.update_image, 0.75)  # Change image every 0.75 seconds
        self.schedule_next_insult()

        if coin_acceptor is not None:
            coin_acceptor.when_activated = self.stop_animation_and_schedule_switch
        else:
            # simulate coin insertion after 5 seconds on non-Pi systems
            Timer(5.0, self.stop_animation_and_schedule_switch).start()

    def schedule_next_insult(self):
        now = time.localtime()
        mins_past = now.tm_min % 30
        secs_until = (30 - mins_past) * 60 - now.tm_sec
        if secs_until <= 0:
            secs_until = 1800
        self.insult_event = Clock.schedule_once(self.speak_random_insult, secs_until)

    def speak_random_insult(self, dt):
        if not self.current_insults:
            self.current_insults = list(self.insult_list)
            random.shuffle(self.current_insults)
            # If the next insult (last in list due to pop()) is the same as the previous one, swap it
            if self.current_insults[-1] == self.last_insult and len(self.current_insults) > 1:
                self.current_insults[-1], self.current_insults[0] = self.current_insults[0], self.current_insults[-1]

        insult = self.current_insults.pop()
        self.last_insult = insult
        speak(insult)
        self.schedule_next_insult()

    def update_image(self, dt):
        if self.current_image == 1:
            self.image_widget.source = f'{path}insertcoin1.png'
            self.current_image = 2
        else:
            self.image_widget.source = f'{path}insertcoin0.png'
            self.current_image = 1

    def stop_animation_and_schedule_switch(self, channel=None):
        self.animation_event.cancel()
        if hasattr(self, 'insult_event'):
            self.insult_event.cancel()
        print("coin received!")
        speak("Coin received. Choose your degradation.")
        Clock.schedule_once(lambda dt: self.switch_to_category_screen(), 0)

    @mainthread
    def switch_to_category_screen(self):
        App.get_running_app().sm.current = 'category'
class InsultMasterApp(App):
    def build(self):
        self.theme_colors = {
            'primary': (201/255, 211/255, 45/255, 1),         	# Lime
            'secondary': (236/255, 109/255, 42/255, 1),       	# Orange
            'accent': (236/255, 109/255, 42/255, 1),          	# Orange
            'background': (39/255, 79/255, 139/255, 1)  		# Blue
        }
        self.theme_vars = {
            'padding': 10,
            'spacing': 10
        }
        self.fonts = {
            'heading': 'FreeMono',
            'body': 'FreeMono',
            'button': 'FreeMono'
        }

        self.sm = ScreenManager(transition=NoTransition())
        self.teach_submission = {}
        self.image_job = ImageJob()
        self.sm.add_widget(SplashScreen(name='splash'))
        self.sm.add_widget(CategoryScreen(name='category'))
        self.sm.add_widget(InsultScreen(name='insult'))
        self.sm.add_widget(CameraScreen(name='camera'))
        self.sm.add_widget(LoadScreen(name='load'))
        self.sm.add_widget(DisplayScreen(name='display'))
        self.sm.add_widget(TeachCategoryScreen(name='teach_category'))
        self.sm.add_widget(TeachAdjScreen(name='teach_adj'))
        self.sm.add_widget(TeachNounScreen(name='teach_noun'))

        return self.sm

    def go_to_category_screen(self, *args):
        self.sm.current = 'category'

    def on_start(self):
        super(InsultMasterApp, self).on_start()
        self.populate_category_buttons()

    def populate_category_buttons(self, *args):
        category_screen = self.sm.get_screen('category')
        for code, label in CATEGORY_DISPLAY:
            btn = ThemedButton(text=label, size_hint_y=None, height=40)
            btn.bind(on_release=lambda instance, c=code: category_screen.select_category(c))
            category_screen.ids.categories.add_widget(btn)

    def on_stop(self):
        # Clean up GPIO so the pin isn't "busy" on the next run
        if coin_acceptor is not None:
            try:
                coin_acceptor.close()
            except Exception as e:
                print(f"Error closing coin_acceptor: {e}")
        return super().on_stop()


if __name__ == '__main__':
    InsultMasterApp().run()
