import tkinter as tk


class MovementAnimation:
    """
    This class represents on a canvas the movement of particles according to different models.

    :param cl: model that is represented in the canvas
    :type cl: class
    :param side: length of the square canvas
    :type side: float
    """

    def __init__(self, cl, side):
        self.cl = cl
        self.side, real_side = side, self.cl.get_side()
        self.ratio = self.side / real_side
        self.radius = self.cl.get_radius() * self.ratio
        self.window = tk.Tk()
        button = tk.Button(self.window, text="X", command=self.window.destroy)
        button.pack()
        self.canvas = tk.Canvas(self.window, width= self.side, height=self.side, bg='white')
        self.canvas.pack()
        self.animation_movement()
        self.window.mainloop()

    def animation_movement(self):
        """
        This function is the one that draws at each iteration the new position of the particles.
        """
        position_array = self.cl.get_position()
        self.canvas.delete("all")
        for i, elt in enumerate(position_array):
            x = elt[0] * self.ratio
            y = elt[1] * self.ratio
            self.canvas.create_oval(x - self.radius, y + self.radius, x + self.radius, y - self.radius, fill='blue')

        self.window.update()
        self.cl.iter_movement(1, animation=True)
        self.window.after(10, self.animation_movement)
