from math import ceil

import numpy as np
import PIL.Image

from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
from smqtk.algorithms.relevancy_index import get_relevancy_index_impls
from smqtk.representation import DescriptorElementFactory
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.utils import plugin


__author__ = 'paul.tunison@kitware.com'


"""
Currently this requires some additional modifications in the ColorDescriptor
impl to expose low-level detections and descriptors. Patch:

diff --git a/python/smqtk/algorithms/descriptor_generator/colordescriptor/colordescriptor.py b/python/smqtk/algorithms/descriptor_generator/colordescriptor/colordescriptor.py
index 8dd7323..28ddd8f 100644
--- a/python/smqtk/algorithms/descriptor_generator/colordescriptor/colordescriptor.py
+++ b/python/smqtk/algorithms/descriptor_generator/colordescriptor/colordescriptor.py
@@ -448,6 +448,10 @@ class ColorDescriptor_Base (DescriptorGenerator):
                         data.uuid())
         info, descriptors = self._generate_descriptor_matrices({data})

+        # DEBUG: Getting access to low-level detections and descriptors
+        self.detections = info
+        self.descriptors = descriptors
+
         if not self._use_sp:
             ###
             # Codebook Quantization


Also, using a RBF kernel with libSVM looks kinda ok. Use of HIK looks sparse.

"""


# Configurations
descriptor_generator_config = {
    "type": "ColorDescriptor_Image_csift",
    "ColorDescriptor_Image_csift": {
        "model_directory": "/Users/purg/dev/smqtk/source/data/ContentDescriptors/ColorDescriptor/csift/example_image",
        "work_directory": "/Users/purg/dev/smqtk/source/work/ContentDescriptors/ColorDescriptor/csift/example_image",
    }
}

descriptor_factory_config = {
    "type": "DescriptorMemoryElement",
    "DescriptorMemoryElement": {},
}

relevancy_index_config = {
    "type": "LibSvmHikRelevancyIndex",
    "LibSvmHikRelevancyIndex": {}
}

# Initialize algorithms
#: :type: smqtk.algorithms.descriptor_generator.DescriptorGenerator
dg = plugin.from_plugin_config(descriptor_generator_config,
                               get_descriptor_generator_impls)
descr_factory = DescriptorElementFactory.from_config(descriptor_factory_config)
#: :type: ()->smqtk.algorithms.relevancy_index.RelevancyIndex
make_r_index = lambda: plugin.from_plugin_config(relevancy_index_config,
                                                 get_relevancy_index_impls)

# Take two images
img1_path = "/Users/purg/data/smqtk/image_sets/example_image_set/F16/F-16_June_2008.jpg"
# img1_path = "/Users/purg/dev/smqtk/source/python/smqtk/tests/data/Lenna.png"
#: :type: PIL.Image.Image
img1 = PIL.Image.open(img1_path)
img1_data = DataFileElement(img1_path)
img1_descr = dg.compute_descriptor(img1_data, descr_factory, True)
img1_hist = img1_descr.vector()
img1_ll_dets = dg.detections
img1_ll_descrs = dg.descriptors

img2_path = "/Users/purg/data/smqtk/image_sets/example_image_set/F16/f16.1.jpg"
# img2_path = "/Users/purg/dev/smqtk/source/python/smqtk/tests/data/beautiful-face.jpg"
#: :type: PIL.Image.Image
img2 = PIL.Image.open(img2_path)
img2_data = DataFileElement(img2_path)
img2_descr = dg.compute_descriptor(img2_data, descr_factory, True)
img2_hist = img2_descr.vector()
img2_ll_dets = dg.detections
img2_ll_descrs = dg.descriptors


# Determine which bins to use for positive and negative relevancy exemplars
# which bins in histogram are non-zero in both histograms
i = (img1_hist > 0) & (img2_hist > 0)
# indices of intersecting bins
i_indices = np.array([j for j, b in enumerate(i) if b])

# intersecting bin histogram distances
# # -> lower value == greater intersection
# hi = np.abs(img1_hist[i] - img2_hist[i])
# # Ordering bin indices where high intersection bins are closer to index 0
# ordered = sorted(zip(hi, i_indices), reverse=1)

# -> high value == greater combined relevance
prod = np.multiply(img1_hist[i], img2_hist[i])
# Ordering bin indices where high products mean
ordered = sorted(zip(prod, i_indices), reverse=1)

ordered_indices = [e[1] for e in ordered]

# Pick number of bins to take as positive and negative exemplars
top_n_perc = 0.05
top_n = int(img1_hist.size * top_n_perc)
if not top_n:
    raise ValueError("Top-n chosen is 0. Adjust top_n_perc to be larger.")
pos_bins = ordered_indices[:top_n]
neg_bins = ordered_indices[-top_n:]
count = 0
pos_cb_descrs = []
for b in pos_bins:
    d = descr_factory.new_descriptor('codebook', count)
    d.set_vector(dg._codebook[b])
    pos_cb_descrs.append(d)
    count += 1
neg_cb_descrs = []
for b in neg_bins:
    d = descr_factory.new_descriptor('codebook', count)
    d.set_vector(dg._codebook[b])
    neg_cb_descrs.append(d)
    count += 1

# Build relevancy indices
img1_descr2det = {}
for r in xrange(img1_ll_descrs.shape[0]):
    d = descr_factory.new_descriptor('img1_descr', r)
    d.set_vector(img1_ll_descrs[r])
    img1_descr2det[d] = img1_ll_dets[r]
img1_ri = make_r_index()
img1_ri.build_index(img1_descr2det.keys())
img1_descr_rank = img1_ri.rank(pos_cb_descrs, neg_cb_descrs)

img2_descr2det = {}
for r in xrange(img2_ll_descrs.shape[0]):
    d = descr_factory.new_descriptor('img2_descr', r)
    d.set_vector(img2_ll_descrs[r])
    img2_descr2det[d] = img2_ll_dets[r]
img2_ri = make_r_index()
img2_ri.build_index(img2_descr2det.keys())
img2_descr_rank = img2_ri.rank(pos_cb_descrs, neg_cb_descrs)


# Create some heatmap
import heatmap
mult_factor = 10.

# Create list of points, repeating detection coordinates depending on associated
# descriptor weight. Rescale float based on range
i1_min = min(img1_descr_rank.values())
i1_max = max(img1_descr_rank.values())
i1_range = i1_max - i1_min
i1_hm_points = []
for descr, r in img1_descr_rank.iteritems():
    s=(r-i1_min)/i1_range
    for j in xrange(int(ceil(s*mult_factor))):
        i1_hm_points.append(img1_descr2det[descr][:2].tolist())
        i1_hm_points[-1][1] = img1.size[1] - i1_hm_points[-1][1] - 1

img1_hm = heatmap.Heatmap()
img1_hm_img = img1_hm.heatmap(
    points=i1_hm_points,
    # dense-sample spacing equation
    dotsize=max(int(min(img1.size) / 50.0), 6) * 3,
    size=img1.size,
    scheme="kitware",
    scale_density=True,
)

i2_min = min(img2_descr_rank.values())
i2_max = max(img2_descr_rank.values())
i2_range = i2_max - i2_min
i2_hm_points = []
for descr, r in img2_descr_rank.iteritems():
    s = (r-i2_min)/i2_range
    for j in xrange(int(ceil(s*mult_factor))):
        i2_hm_points.append(img2_descr2det[descr][:2].tolist())
        i2_hm_points[-1][1] = img2.size[1] - i2_hm_points[-1][1] - 1

img2_hm = heatmap.Heatmap()
img2_hm_img = img2_hm.heatmap(
    points=i2_hm_points,
    # dense-sample spacing equation
    dotsize=max(int(min(img2.size) / 50.0), 6) * 3,
    size=img2.size,
    scheme="kitware",
    scale_density=True,
)


# Overlay images
img1.paste(img1_hm_img, (0,0), img1_hm_img)
img2.paste(img2_hm_img, (0,0), img2_hm_img)
