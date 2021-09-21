import tkinter as tk
import numpy as np


class MovementAnimation:
    """
    This class represents on a canvas the movement of particles according to different models.

    :param cl: model that is represented in the canvas
    :type cl: class
    :param side: length of the square canvas
    :type side: float
    """

    def __init__(self, cl, side=1000):
        self.cl = cl
        self.side, real_side = side, self.cl.get_side()
        self.ratio = self.side / real_side
        self.radius = self.cl.get_radius() * self.ratio
        self.janus = self.cl.get_janus()
        self.step = 0
        self.window = tk.Tk()
        button = tk.Button(self.window, text="X", command=self.window.destroy)
        button.pack()
        self.canvas = tk.Canvas(self.window, width=self.side, height=self.side, bg='white')
        self.canvas.pack()
        self.animation_movement()
        self.window.mainloop()

    def animation_movement(self):
        """
        This function is the one that draws at each iteration the new position of the particles.
        """
        position_array = self.cl.get_position()
        #velocities_array = self.cl.get_velocities()

        if self.janus:
            velocities_array = self.cl.get_velocities()
            #a=1
        else:
            self.canvas.delete("all")

        for i, elt in enumerate(position_array):
            x, y = elt[0] * self.ratio, elt[1] * self.ratio
            #v = velocities_array[i]
            if self.janus:
                v = velocities_array[i]

                if np.any(v):
                    self.canvas.delete(str(i)+'a')
                    v_per = np.array([-v[1], v[0]])
                    angle = 360 - np.angle(np.complex(v_per[0], v_per[1]), deg=True)

                    if angle < 0:
                        angle += 360

                    self.canvas.create_arc(x - self.radius, y + self.radius, x + self.radius, y - self.radius,
                                           start=angle, extent=180, fill='blue', tag=str(i)+'a')

                    if angle < 180:
                        new_angle = 180 + angle

                    else:
                        new_angle = angle - 180

                    self.canvas.create_arc(x - self.radius, y + self.radius, x + self.radius, y - self.radius,
                                           start=new_angle, extent=180, fill='red', tag=str(i)+'a')

            else:
                self.canvas.create_oval(x - self.radius, y + self.radius, x + self.radius, y - self.radius, fill='blue')
                #self.canvas.create_text(x-self.radius/2, y, text=str(i), fill='red')
                #target = self.cl.get_target()
                #self.canvas.create_text(x+self.radius/2, y, text=str(target[i]), fill='green')
                #self.canvas.create_text(x + 3 * self.radius, y, text=str(np.linalg.norm(v))[:6], fill='green')
        self.step += 1
        self.window.update()
        self.cl.iter_movement(self.step, animation=True)
        self.window.after(10, self.animation_movement)
