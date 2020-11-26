# Clock

## Algorithm:

### The first version:
1. Cut the image by a clock face:
    1. Find the clock face;
    2. Compute its width and height;
    3. Cut it by width and height.
2. Define a center of a clock face:
    1. Find digits;
    2. Build a horizontal line that runs through two opposite digits;
    *3. Approximate Contours;*
    4. Define a center by it;
    5. Compute angles' shift by it.
3. Find hands of the clock face:
    1. Find all lines on the clock face;
    2. Sort lines by distance between them and the clock face's center;
    3. Compute angles between the clock's sorted hands and the horizontal line;
    4. Add the shift to angles;
4. Compute time by angles.

### The second version:
1. Detecting changes in the video:
    1. Find changes contours and save them like hands of the clock;
2. Try to find saved contours in a static images of the clock;

## Targets
1. Define hands by templates on an image [+]
2. Define a line by a template [+]
3. Define time by lines [-]

## References
1. [Approx](https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html)
2. [Find center](https://issue.life/questions/49068444)