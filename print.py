from escpos.printer import Usb

p = Usb(0x0416, 0x5011, in_ep=0x81, out_ep=0x01, profile='POS-5890')
p.text("Hello from the Pi!\n")
p.cut()
