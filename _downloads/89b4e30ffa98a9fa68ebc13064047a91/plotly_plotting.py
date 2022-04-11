"""
.. _plotly-plotting:

====================
Show Plotly plotting
====================

This is just the default sphinx-gallery plotly example,
to test how this works.
"""
# %%
import plotly.express as px
import numpy as np

df = px.data.tips()
fig = px.bar(df, x='sex', y='total_bill', facet_col='day', color='smoker', barmode='group',
             template='presentation+plotly'
             )
fig.update_layout(height=400)
fig

# %%
# Create Thumbnail for the Sphinx-Gallery:
from pathlib import Path 
if Path("../docs/gallery").exists():
    fig.write_image("../docs/gallery/_thumb_plotly_plotting.png")
# sphinx_gallery_thumbnail_path = 'gallery/_thumb_plotly_plotting.png'
# %%
