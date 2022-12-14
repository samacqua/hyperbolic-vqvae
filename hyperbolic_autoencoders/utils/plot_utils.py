import matplotlib.pyplot as plt

def getfig(figsize, xmargin=0.18, ymargin=0.2, xoffset=0, yoffset=0):
    width, height = figsize
    fig = plt.figure(figsize=(
        width / (1 - xmargin * 2), height / (1 - ymargin * 2)
    ))
    fig.subplots_adjust(
        left=(xmargin + xoffset), right=(1 - xmargin + xoffset),
        bottom=(ymargin + yoffset), top=(1 - ymargin + yoffset))
    return fig
