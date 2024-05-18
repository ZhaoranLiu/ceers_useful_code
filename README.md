MyGrism

A tool for reducing and analyzing grism data.

Demonstration
~~~~~~~~~~~~~
.. image:: ./demo.gif
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install mygrism

or

.. code-block:: bash

    python setup.py install

Usage
~~~~~

**Please set your CRDS path in the core.py first:**

.. code-block:: python

    import os

    os.environ['CRDS_PATH'] = '…'

## If you want to do within 5 steps

.. code-block:: bash

    mygrism super_background --data_type GRISMR
    mygrism cont_subtraction --L_box 25 --L_mask 4
    mygrism reduce_sw
    mygrism spectra_extration
    mygrism plot_spectra

## After you finished the first 3 steps, you can easily change the catalog in config.py and only rerun the ‘spectra_extration’ and ‘plot_spectra’

## If you want to do the reduction step by step
Please check out the cli.py to see the commands:
.. code-block:: bash

    mygrism --help

## Produced background-subtracted images and continuum-subtracted images:
.. code-block:: bash

    ./grism_cal/plots

## Please make sure the project structure is:
.. code-block:: bash

    xxx/
    ├── my_grism/
    │   ├── __init__.py
    │   ├── core.py
    │   ├── config.py
    │   ├── cli.py
    │   └── data/
    │       └── GRISM_NIRCAM/
    │       └── FSun_cal/
    │           ├── FSun_SpecCov_*.fits
    │           └── disper/
    │           └── sensitivity/
