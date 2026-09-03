# Private input schema

The canonical scripts expect one row per indexed PCM-TES scenario state. Column names are case-sensitive.

## Identifiers and context

| Field | Type or unit | Analysis role |
|---|---|---|
| `timestamp` | parseable datetime | chronological splitting and calendar features |
| `pcm_type` | category | material component of the action |
| `system_type` | category | application component of the action |
| `encapsulation_type` | category | packaging component of the action |
| `air_temperature_c` | degC | pre-decision context |
| `relative_humidity_pct` | % | pre-decision context |
| `wind_speed_mps` | m/s | pre-decision context |
| `cloud_cover_pct` | % | pre-decision context |
| `solar_irradiance_wm2` | W/m2 | pre-decision context |
| `inlet_fluid_temp_c` | degC | pre-decision context |
| `cycle_number` | count | pre-decision context and degradation history |

## Thermophysical, geometric, and process fields

| Field | Unit |
|---|---|
| `melting_point_c` | degC |
| `latent_heat_kjkg` | kJ/kg |
| `thermal_conductivity_wmk` | W/(m K) |
| `density_kgm3` | kg/m3 |
| `specific_heat_jkgk` | J/(kg K) |
| `pcm_mass_kg` | kg |
| `surface_area_m2` | m2 |
| `pcm_thickness_mm` | mm |
| `mass_flow_rate_kgs` | kg/s |
| `degradation_factor` | dimensionless |
| `temp_difference_c` | degC |
| `phase_fraction` | 0-1 |
| `heat_transfer_coeff_wm2k` | W/(m2 K) |
| `heat_flux_wm2` | W/m2 |

## Performance fields

| Field | Unit or scale |
|---|---|
| `stored_energy_kj` | kJ |
| `energy_input_kj` | kJ |
| `charging_time_min` | min |
| `discharging_time_min` | min |
| `energy_loss_pct` | % |
| `state_of_charge_pct` | % |
| `cooling_load_offset_pct` | % |
| `thermal_storage_efficiency_pct` | % |

## Constructed decision criteria

The code constructs storage density, areal storage density, charge and discharge power, reversed loss, and reversed response-time scores. These are deliberately correlated engineering views of the same storage process. EFA is used to model their shared variance before AHP weighting.

The schema documents the software interface; it does not establish record-level provenance or validate the physical origin of any supplied value.

