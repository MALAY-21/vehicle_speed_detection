<h1>Vehicle Speed Detection </h1>
<ul>
<li>This Python script is designed to detect the speed of multiple vehicles on multi-lane highways while also incorporating number plate detection if a vehicle is overspeeding. The script utilizes a Haar Cascade Classifier to detect vehicles in every nth frame, optimizing processing speed by removing unnecessary portions from the image. Two reference lines are set for vehicle entry and exit.

When a vehicle crosses the entry point, the time is recorded, and the vehicle is tracked using centroid tracking techniques. The time is recorded again when the vehicle crosses the exit line, and based on the time difference, the vehicle's speed is estimated.</li>
</ul>

<h1>Additional Feature: Number Plate Detection for Overspeeding Vehicles</h4>
<ul>
<li>If a vehicle is detected to be overspeeding, an additional step is taken to detect its number plate. This information can be valuable for law enforcement or monitoring purposes. The script utilizes a number plate detection algorithm to identify and record the number plate details of the overspeeding vehicle.</li>
</ul>
