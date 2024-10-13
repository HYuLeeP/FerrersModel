# Ferrers Model

This is a github repo for the MilkyWay Bar model with a slow, long, Ferrers bar (based on results from Wegg et al. (2015)).

In this project, you can find two .py modules FerrersModel and HYuPlot with classes and methods helpful for orbit integration and plotting.

The potential and potential gradient with a rotating Ferrers bar and a logarithmic disk/bulge/halo is written in kcf_L4_mod.py. some configurations are stored in config. (This is currently a mess, so try no to change anything)

A basic tutorial is provided in Tutorial.ipynb.

More details of the methods and previous results using this model can be found at [this draft paper](https://www.mso.anu.edu.au/~lyusen/hercules_draft.pdf).

Contact kenneth.freeman@anu.edu.au or li.yusen.astr@gmail.com to reach out.

To include kinematics data from GALAH, download \ref{https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/galah_dr4_vac_dynamics_240705.fits}{the dynamics VAC} and put it under ./data/.