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
#
# .. tip::
#       Some Tip about how to use this.     
#       Also, a reference, because why not :cite:`TomasAmplitudeDependentClosest2016`.

# %%
# Create Thumbnail for Gallery:
fig.write_image("../docs/gallery/_thumb_plotly_plotting.png")
# sphinx_gallery_thumbnail_path = 'gallery/_thumb_plotly_plotting.png'
# %%
