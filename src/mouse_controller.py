
import pyautogui


class MouseController:
    def __init__(self, precision, speed):
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':0.5, 'slow':10, 'medium':5}
        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):
        offset_x, offset_y = x*self.precision, -1*y*self.precision
        if self._on_screen(offset_x, offset_y):
            pyautogui.moveRel(offset_x, offset_y, duration=self.speed)
    
    def _on_screen(self, offset_x, offset_y):
        current_x, current_y = pyautogui.position()
        return pyautogui.onScreen(current_x + offset_x, current_y + offset_y)
