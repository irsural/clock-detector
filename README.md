# Clock

## Algorithms

### The general part of #1 & #2 algorithms
1. Get a frame from a camera capture
2. Find the watch face using the template search (SIFT)

### By analize a graph of an image #1
3. Rotate the watch face to the polar system
4. Translate the found area to grayscale, then get threshold
5. Convert the filtered image to graph by counting all no-black pixels
6. Find extremes of the graph and sort them removing useless ones
7. The remaining extremes after filtering will be the hands of the clock

### By finding the lines out #2
3. Translate the found area of grayscale, then get threshold
4. Find lines by cv2.HoughLinesP
5. Filter the found lines by computing distance from them to the center of the watch face
6. Compute the angle between lines relative to the watch face

## todo:
1. Remove highlights on a photo

## References
1. [Wrap Polar](https://www.programmersought.com/article/1787117182/)