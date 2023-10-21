# coding=utf-8
# Copyright 2023 Ling-Hao CHEN (https://lhchen.top) from Tsinghua University。
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, you need to cite as:
#
# @inproceedings{chen2023humanmac,
# 	title={HumanMAC: Masked Motion Completion for Human Motion Prediction},
# 	author={Chen, Ling-Hao and Zhang, Jiawei and Li, Yewen and Pang, Yiren and Xia, Xiaobo and Liu, Tongliang},
# 	journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
# 	year={2023}
# }
#
# If you use the code, you need to declare the copyright!
#
# Licensed under the MIT License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# retargeting skeleton from H36M to AMASS
# refer to https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing


# Load the 'amass.npy' dataset using NumPy.
fulldata = np.load('amass.npy', allow_pickle=True)
# Create an array of zeros to represent the root position for the retargeted data.
root = np.zeros((5000, 160, 1, 1, 3))
# Add the root position to the 'fulldata' by concatenating it along the second-to-last dimension.
fulldata = np.concatenate((root, fulldata), axis=-2)
# Create a new array 'new_data' with the same shape as 'fulldata' but with a different skeleton structure.
new_data = np.zeros_like(fulldata)[:, :, :, :17, :]

# Map the joints from the H36M skeleton to the new structure in 'new_data'.
# This mapping aligns the joints to match the new skeleton structure.
new_data[:, :, :, 0, :] = fulldata[:, :, :, 0, :]
new_data[:, :, :, 1, :] = fulldata[:, :, :, 1, :]
new_data[:, :, :, 2, :] = fulldata[:, :, :, 4, :]
new_data[:, :, :, 3, :] = fulldata[:, :, :, 7, :]
new_data[:, :, :, 4, :] = fulldata[:, :, :, 2, :]
new_data[:, :, :, 5, :] = fulldata[:, :, :, 5, :]
new_data[:, :, :, 6, :] = fulldata[:, :, :, 8, :]
new_data[:, :, :, 7, :] = (fulldata[:, :, :, 3, :] +
                           fulldata[:, :, :, 6, :]) / 2
new_data[:, :, :, 8, :] = fulldata[:, :, :, 9, :]
new_data[:, :, :, 9, :] = fulldata[:, :, :, 12, :]
new_data[:, :, :, 10, :] = fulldata[:, :, :, 15, :]
new_data[:, :, :, 11, :] = fulldata[:, :, :, 17, :]
new_data[:, :, :, 12, :] = fulldata[:, :, :, 19, :]
new_data[:, :, :, 13, :] = fulldata[:, :, :, 21, :]
new_data[:, :, :, 14, :] = fulldata[:, :, :, 16, :]
new_data[:, :, :, 15, :] = fulldata[:, :, :, 18, :]
new_data[:, :, :, 16, :] = fulldata[:, :, :, 20, :]


# Print the shapes of 'fulldata' and 'new_data'.
print(fulldata.shape)
print(new_data.shape)

# Save the 'new_data' as 'retargeted.npy' for future use.
np.save("retargeted.npy", new_data)

# Extract and print data for a specific example (sample 66) from the original 'fulldata'.
data = np.squeeze(fulldata[66], axis=1)
print(data.shape)
