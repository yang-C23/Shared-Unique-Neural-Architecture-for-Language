import numpy as np
from matplotlib import pyplot as plt
from surfplot import Plot


def build_surf(
    plot: Plot,
    figure=None,
    axis=None,
    cbar: bool = True,
    cbar_location: str = "right",
    cbar_aspect: int = 8,
    **kwargs
):
    if figure is None and axis is None:
        return plot.build(**kwargs)

    # copied from source code of Plot.build() so i can use my own fig/ax.
    plotter = plot.render()
    plotter._check_offscreen()
    x = plotter.to_numpy(transparent_bg=True, scale=(2, 2))

    if axis is None:
        figsize = tuple((np.array(plot.size) / 100) + 1)
        axis = figure.subplots(figsize=figsize)

    axis.imshow(x)
    axis.axis("off")

    if cbar:
        plot._add_colorbars(
            ax=axis,
            n_ticks=2,
            location=cbar_location,
            aspect=cbar_aspect,
            fontsize=plt.rcParams.get("xtick.labelsize", 8),
        )

    return figure
