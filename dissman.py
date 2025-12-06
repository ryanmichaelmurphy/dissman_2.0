from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.lang import Builder
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from kivy.clock import mainthread
import random
import io
import threading
import requests
import time
import os
import subprocess
import cv2
import numpy as np
from gpiozero import Button as GPIOButton
#from signal import pause
from escpos.printer import Usb
from openai import OpenAI

GPIO_PIN = 17
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


coin_acceptor = GPIOButton(GPIO_PIN)

p = Usb(0x0416, 0x5011, in_ep=0x81, out_ep=0x01, profile='POS-5890')

path = '/home/dissman/Documents/app/' # path to files 

Builder.load_file(path + 'insultmaster3.kv')

Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '480')
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'fullscreen', '1')

# G-rated adjectives and nouns
g_adj = "clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy"
g_nouns = "doodle hamburger backpack bedding bedspread binder blanket blinds bookcase book broom brush bucket calendar angler toad horse candle carpet chair china clock coffee-table comb comforter computer container couch credenza curtain cushion heater houseplant magnet mop radiator radio refrigerator rug saucer saw scissors screwdriver settee shade sheet shelf shirt shoe smoke-detector sneaker socks sofa speaker toy tool tv toothpaste towel nutcase toaster pancake muffin wombat caboose goblin pirate ninja meatball cupcake tadpole dingbat noodle turnip alien gadget grasshopper pickle wigwam bonnethead sharksucker"

# R-rated adjectives and nouns (sensitive content redacted)
r_adj = "shitty great-value pick-me horsefaced cum-guzzling christian vaginal straight cum semen smegma discharge fallopian anal aggressive arrogant boastful bossy boring careless clingy cruel cowardly deceitful dishonest greedy harsh impatient impulsive jealous moody narrow-minded overcritical rude selfish untrustworthy unhappy cumguzzling unfuckable incestuous sick perverted deranged depraved mountain-dew-drinking butterfaced self-centered revolting repellent repulsive sickening nauseating nauseous stomach-churning stomach-turning off-putting unpalatable unappetizing uninviting unsavoury distasteful foul nasty obnoxious odious"
r_nouns = "eunuch no-dick cum-for-brains toesniffer dicksucker dicksniffer toesucker simp skidmark shit-stain anal-fissure anal-wart anal-cyst vaginal-cyst vaginal-discharge smegma foreskin dick-cheese cumguzzler yuppy hippy karen boomer dork nerd dweeb unfuckable pedophile butterface needledick incel neckbeard wart genital-wart homewrecker doof douche doucheholster douchebag cum-receptacle cum-dumpster cumrag cumslut bum degenerate derelict good-for-nothing no-account no-good slacker hetrosexual buttlover breitbart-reader andriod-user trump-lover republican cumslut buttmuncher nutsack ballsack boner christian penis cunt twat asshole fucker shitbag shit-for-brains cumrag gland intestine cecum colon rectum liver gallbladder mesentery pancreas anus kidney ureter bladder urethra ovary tube uterus cervix discharge vagina"

# Old-timey adjectives and nouns
old_adj = "froward pernickety laggardly moonstruck mumpsimus spleeny fribble dandiprat rattlecap slugabed cacafuego raggabrash dithering muddle-headed tatterdemalion claptrap wifty bedswerver lackadaisical flapdoodle"
old_nouns = "scallywag naysayer neerdowell landlover fustilarian snollygoster popinjay lickspittle rakefire whippersnapper noodle mumblecrust zounderkite gillywetfoot doodle pettifogger fopdoodle mooncalf clodpole hugger-mugger ragamuffin scalawag ninnyhammer flapdragon"

# Combine to make an all list
all_adj = g_adj + r_adj + old_adj
all_nouns = g_nouns + r_nouns + old_nouns

# Convert to lists
g_adj_list = g_adj.split(" ")
g_noun_list = g_nouns.split(" ")
r_adj_list = r_adj.split(" ")
r_noun_list = r_nouns.split(" ")
old_adj_list = old_adj.split(" ")
old_noun_list = old_nouns.split(" ")
all_adj_list = all_adj.split(" ")
all_noun_list = all_nouns.split(" ")

categories = {
    "G-rated": (g_adj_list, g_noun_list),
    "R-rated": (r_adj_list, r_noun_list),
    "Old-timey": (old_adj_list, old_noun_list),
    "Anything Goes": (all_adj_list, all_noun_list),
}

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
    def generate_insults(self, adj_list, noun_list):
        return [f"{random.choice(adj_list)} {random.choice(noun_list)}" for _ in range(3)]

    def on_enter(self, *args):
        self.ids.insult_options.clear_widgets()  # Clear existing buttons
        category = self.manager.current_category
        adj_list, noun_list = categories[category]
        insults = self.generate_insults(adj_list, noun_list)
        for insult in insults:
            btn = ThemedButton(text=insult, size_hint_y=None, height=40)
            btn.bind(on_release=self.show_insult)
            self.ids.insult_options.add_widget(btn)
        self.ids.header.text = "What best describes you?"

    def show_insult(self, instance):
        article = "an" if instance.text[0] in 'aeiou' else "a"
        self.manager.transition.direction = 'left'
        self.manager.current = 'camera' 
        self.manager.get_screen('display').ids.insult_label.text = f"you {instance.text}."

class CameraScreen(Screen):
    def on_enter(self):
        # Clear any previous image or terminal display
        os.system('clear')  # This clears the terminal screen
        self.setup_camera()
        self.img1 = Image()
        self.add_widget(self.img1)
        
        # Add overlay text
        self.overlay_text = Label(
            text="Let me get a good look at you",
            size_hint=(None, None),
            size=(400, 50),  # Adjust size as needed
            pos_hint={'center_x': 0.5, 'top': 1},
            font_name='FreeMono',  # Change to desired font
            font_size='24sp',    # Change to desired font size
            color= [0,0,0]
            )
        self.add_widget(self.overlay_text)

                
        # Start webcam preview
        Clock.schedule_interval(self.update_preview, 1 / 30)  # ~30fps for smoother preview
        # Schedule taking the picture after 3 seconds
        Clock.schedule_once(self.capture_image, 3)

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
        # Ensure a frame is available
        ret, frame = self.camera.read()
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=50)
        if ret:
            timestamp = str(int(time.time()))
            save_path = path + 'test_' + timestamp + '.png'
            cv2.imwrite(save_path, frame) 

            # Store the save_path in the App class
            App.get_running_app().last_image_path = save_path

            # Update the image on the screen
            self.ids.captured_image.source = save_path

            # Stop the preview process and release the camera
            Clock.unschedule(self.update_preview)
            if self.camera.isOpened():
                self.camera.release()

            # Schedule showing the loading screen after 2 seconds
            Clock.schedule_once(self.show_loading_screen, 2)

    def show_loading_screen(self, dt):
        self.manager.current = 'load'

class LoadScreen(Screen):
    def on_enter(self, *args):
        self.timestamp = str(int(time.time()))  # Get current timestamp
        self.image_path = f'{path}downloaded_image_{self.timestamp}.png'  # Class attribute
        self.image_ready = False
        self.old_image_url = None
        self.image_widget = Image(source=f'{path}thinking0.png')
        self.add_widget(self.image_widget)
        self.current_image = 1
        threading.Thread(target=self.fetch_image).start()
        Clock.schedule_interval(self.check_image_ready, 0.3)

    def fetch_image(self):
        try:
            # Get the last image path from the App class
            last_image_path = App.get_running_app().last_image_path

            with open(last_image_path, "rb") as image_file:
                response = client.images.create_variation(
                    model="dall-e-2",
                    image=image_file,
                    n=1,
                    size="1024x1024"
                )

            response_dict = response.to_dict()
            new_image_url = response_dict['data'][0]['url']
            image_response = requests.get(new_image_url)
            
            if image_response.status_code == 200:
                with open(self.image_path, 'wb') as file:
                    file.write(image_response.content)
                self.image_ready = True  # Update image readiness
                self.old_image_url = new_image_url  # Update the old image URL
            else:
                print(f"Error: Received status code {image_response.status_code} from image request.")
        except requests.RequestException as e:
            print(f"Error: Failed to fetch the image from the URL. {e}")
            Clock.schedule_once(
                lambda dt: setattr(self.manager, "current", "splash"),
                0,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            Clock.schedule_once(
                lambda dt: setattr(self.manager, "current", "splash"),
                0,
            )

    def check_image_ready(self, dt):
        base_path = f'{path}thinking'
        num_images = 16
        images = [f'{base_path}{i}.png' for i in range(num_images)]
        
        self.current_image = (self.current_image + 1) % len(images)
        self.image_widget.source = images[self.current_image]
        
        if self.image_ready and self.old_image_url:  # Check if the image is ready and URL has changed
            self.manager.get_screen('display').ids.dall_e_image.source = self.image_path
            self.manager.current = 'display'
            self.image_ready = False  # Reset the image readiness
            return False  # Unschedule the interval if image is ready

class DisplayScreen(Screen):
    def on_enter(self, *args):
        # Add functionality to print image and insult text
        self.ids.qr_button.clear_widgets()  # Clear existing buttons
        dall_e_image_path = self.ids.dall_e_image.source
        insult_text = self.ids.insult_label.text
        self.print_image_and_text(dall_e_image_path, insult_text, p)

        self.ids.qr_button.size_hint = (1, None)
        self.ids.qr_button.height = 40
        qr_button = ThemedButton(text='See the code and "artist" statement', size_hint_y=None, height=40)
        qr_button.bind(on_release=lambda x: self.print_qr(p))
        self.ids.qr_button.add_widget(qr_button)

        # Schedule transition to the splash screen after 10 seconds
        Clock.schedule_once(lambda x: self.cleanup_and_restart(), 10)

    def cleanup_and_restart(self):
        self.p = None  # Release the printer by removing the reference
        self.manager.current = 'splash'

    def print_qr(self, p):
        print("DEBUG: print_qr called")
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

class CategoryScreen(Screen):
    def select_category(self, category):
        self.manager.current_category = category
        self.manager.transition.direction = 'left'
        self.manager.current = 'insult'

class SplashScreen(Screen):
    def on_enter(self, *args):
        self.image_widget = Image(source=f'{path}insertcoin0.png')
        self.add_widget(self.image_widget)
        self.current_image = 1
        print("waiting for coin...")
        self.animation_event = Clock.schedule_interval(self.update_image, 0.75)  # Change image every 0.75 seconds
        coin_acceptor.when_activated = self.stop_animation_and_schedule_switch

    def update_image(self, dt):
        if self.current_image == 1:
            self.image_widget.source = f'{path}insertcoin1.png'
            self.current_image = 2
        else:
            self.image_widget.source = f'{path}insertcoin0.png'
            self.current_image = 1

    def stop_animation_and_schedule_switch(self, channel):
        self.animation_event.cancel()
        print("coin received!")
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
        self.sm.add_widget(SplashScreen(name='splash'))
        self.sm.add_widget(CategoryScreen(name='category'))
        self.sm.add_widget(InsultScreen(name='insult'))
        self.sm.add_widget(CameraScreen(name='camera'))
        self.sm.add_widget(LoadScreen(name='load'))
        self.sm.add_widget(DisplayScreen(name='display'))

        return self.sm
    
    def go_to_category_screen(self, *args):
        self.sm.current = 'category'
        
    def on_start(self):
        super(InsultMasterApp, self).on_start()
        self.populate_category_buttons()

    def populate_category_buttons(self, *args):
        category_screen = self.sm.get_screen('category')
        for category in categories.keys():
            btn = ThemedButton(text=category, size_hint_y=None,
            height=40)
            btn.bind(on_release=lambda instance, c=category: category_screen.select_category(c))
            category_screen.ids.categories.add_widget(btn)
        
    def on_stop(self):
        # Clean up GPIO so the pin isn't "busy" on the next run
        try:
            coin_acceptor.close()
        except Exception as e:
            print(f"Error closing coin_acceptor: {e}")
        return super().on_stop()


if __name__ == '__main__':
    InsultMasterApp().run()
