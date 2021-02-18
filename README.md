# JVanalyzer

A simple GUI and code to enable quick plotting and reporting of solar cell's photovoltaic performances. 
Implemented to work on export files from Litos Lite (Fluxim).

The naming convention '<device label> - <variable+value>' should be used when saving device data.  
<device label>: unique identifier for a device in the batch
<variable+value>: the name of the variable under control and of its value.
		  Used in the report of device performance statistics.
		  Devices with same <variable+value> field will be averaged together. 