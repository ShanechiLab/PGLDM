"""Common methods for visualizing data & metrics."""
import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

_FIGSIZE = (8, 4)

def plot_eigenvalues(eigs, title, labels=['true', 'id'],
                     markers=['o', 'x'],
                     edgecolors=['#0000ff', 'None'],
                     facecolors=['None', '#00aa00'],
                     alpha_vals=[0.5, 0.5], bound_lims_to_circle=False,
                     show_legend=True, legend_location='lower_center',
                     fig=None, ax=None):
  """Plots eigenvalues on z-transform unit circle.

  Args:
    eigs: list of lists or arrays. List of all sets of eigenvalues to plot. For
      example: [[eigs1], [eigs2]].
    title: str. Title to use in plot.
    labels: list of str. The label of each eigenvalue set.
    markers: list of str. The marker type to use for each eigenvalue set.
    edgecolors: list of str. The edge color to use for each marker type.
    facecolors: list of str. The color to use for each marker face.
    alpha_values: list of floats. Alpha values to use for each eigenvalue plot.
    bound_lims_to_circle: bool. Will plot the z-transform unit circle bound to
      to the endges of the unit circle. Default False.
    show_legend: bool. Default True.
    legend_location: str. One of the matplotlib supported locations to place the
      plot legend. Default 'lower_center'.
    fig: Optional. matplotlib.figure.Figure. If none provided will automatically
      create a new figure, otherwise will plot traces on provided figure.
    ax: Optional. matplotlib.axes. If none provided will automatically create
      a new figures with axes. Otherwise will plot traces on provided figure.

  Returns:
    Figure with plots.
  """
  if fig is None and ax is None:
    fig = plt.figure(figsize=_FIGSIZE)
    axs = fig.subplots(1, 2)
    axs[1].remove()
    ax = axs[0]

  ax.axis('equal')
  ax.add_patch(patches.Circle((0,0), radius=1, fill=False, color='black',
               alpha=0.2, ls='-') )
  ax.plot([-1,1,0,0,0], [0,0,0,-1,1], color='black', alpha=0.2, ls='-')

  # Adjust alpha length automatically. Gracefully handles if user doesn't input
  # multiple alpha values.
  if len(alpha_vals) != len(eigs):
    for _ in range(len(eigs) - len(alpha_vals)):
      alpha_vals.append(alpha_vals[0])

  for i in range(len(eigs)):
    eig, label, marker = eigs[i], labels[i], markers[i]
    edgecolor, facecolor, alpha = edgecolors[i], facecolors[i], alpha_vals[i]
    ax.scatter(np.real(eig), np.imag(eig), marker=marker, edgecolors=edgecolor,
               facecolors=facecolor, alpha=alpha, label=label)

  if show_legend:
    ax.legend(loc=legend_location)
  ax.set_title(title)
  if bound_lims_to_circle:
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
  return fig
