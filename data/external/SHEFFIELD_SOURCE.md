# External Sheffield reflectance dataset

This repository does **not** redistribute the raw Sheffield measurements. The manuscript's limited real-spectrum diagnostic uses data associated with:

- N. E. Sánchez-Arriaga, D. Tiwari, W. Hutabarat, A. Leyland, and A. Tiwari, “A Spectroscopic Reflectance-Based Low-Cost Thickness Measurement System for Thin Films: Development and Testing,” *Sensors* 23(11), 5326 (2023). DOI: https://doi.org/10.3390/s23115326
- University of Sheffield dataset: **Dataset: A Spectroscopic Reflectance-Based Low-Cost Thickness Measurement System for Thin Films: Development and Testing**. DOI: https://doi.org/10.15131/shef.data.23285603 ; versioned DOI reported by the Sheffield thesis record: https://doi.org/10.15131/shef.data.23285603.v1

The source paper identifies an Avantes Si/SiO2 reference standard (serial number 1908001) with vendor-reported thicknesses 476.3 nm and 198.7 nm. The present study uses the public measurements only as a failure-mode diagnostic; it does not treat the vendor values as traceable reference measurements with stated uncertainty.

The original source study describes spectroscopic reflectometry as comparing coated and uncoated reflected intensities. This is the basis for the manuscript's explicit warning that those relative curves are not semantically identical to the absolute-reflectance forward model used in our blind inversion.
