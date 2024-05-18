# MyGrism

**Please set your CRDS path in the `core.py` first:**

```python
os.environ['CRDS_PATH'] = '…'

**If you want to do within 5 steps:**
mygrism super_background --data_type GRISMR
mygrism cont_subtraction --L_box 25 --L_mask 4
mygrism reduce_sw
mygrism spectra_extration
mygrism plot_spectra
