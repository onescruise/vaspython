import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio
import os


class animation():
    def __init__(self):
        pass

    def readPicture(self, path):
        os.chdir(path)
        self.images = []
        filenames = sorted(
            (fn for fn in os.listdir('.') if fn.endswith('.png')))
        print(filenames)
        for filename in filenames:
            self.images.append(imageio.imread(filename))

    def outputGif(self, path, filename):
        os.chdir(path)
        name = filename + '.gif'
        imageio.mimsave(name, self.images, duration=2)


# for filename in filenames:
#     images.append(imageio.imread(filename))
# imageio.mimsave('gif.gif', images,duration=
if __name__ == "__main__":
    path = '/media/ones/My Passport/2Ddatas/C2DB'
    a = animation()
    a.readPicture(path)
    a.outputGif(path, filename='2D_bilayers_Se')
