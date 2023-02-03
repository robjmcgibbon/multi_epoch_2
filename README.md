# multi_epoch_2

To ignore/unignore config file
`$ git update-index --skip-worktree config.yaml`
`$ git update-index --no-skip-worktree config.yaml`

Python environment 
`$ conda create --name btml python=3.9.2`
`$ conda activate btml`
`$ conda install pip`
`$ pip install -r requirements`

Place virgo credentials in virgo_credentials.yaml, set permissions
`$ chmod 600 virgo_credentials.yaml`
Ignore virgo_credentials file to avoid committing credentials
`$ git update-index --skip-worktree virgo_credentials.yaml`

| Property                  | Units                            |
|---------------------------|----------------------------------|
| Ages                      | Gyr                              |
| Mass                      | M<sub>⊙</sub>                    |
| Black hole accretion, SFR | M<sub>⊙</sub> / yr               |
| Lengths                   | Mpc                              |
| Magnitudes                | mag                              |
| Metallicity               | M<sub>z</sub> / M<sub>tot</sub>  |


