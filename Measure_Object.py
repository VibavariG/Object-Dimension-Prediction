from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2 as cv

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


img = cv.imread("objects1.jpg", 0)
image = cv.imread("objects1.jpg", 1)
cv.imshow('Original Image', image)
cv.waitKey(5000)

gray = cv.GaussianBlur(img, (7, 7), 0)
edged = cv.Canny(gray, 50, 100)

cv.imshow('Canny', edged)
cv.waitKey(5000)

# Finds object from black background
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

print("Coin : 2.4cm, 2.4cm")
print("Pencil : 18.1cm, 0.8cm")
print("Stone : 3.3cm, 2.8cm")
print("Eraser : 3.4cm, 1.7cm")
print("Apple : 5.8cm, 5.8cm")
print("Sharpener : 3.5cm, 1.7cm")

for c in cnts:
	if cv.contourArea(c) < 20:   # Ignoring the small contours
		continue

	orig = image.copy()
	box = cv.minAreaRect(c) 			# Creates boundary around the object
	box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
	box = np.array(box, dtype="int")				# Storing vertices of box around the object in the form of an array

	box = perspective.order_points(box)
	cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)		# Drawing the edges of box on original image

	for (x, y) in box:
		cv.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)			# Drawing the vertices of box on original image

	(tl, tr, br, bl) = box 				# Storing vertices
	# Finding midpoints of all the edges
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# Drawing the midpoints of box on original image
	cv.circle(orig, (int(tltrX), int(tltrY)), 5, (0, 0, 255), -1)
	cv.circle(orig, (int(blbrX), int(blbrY)), 5, (0, 0, 255), -1)
	cv.circle(orig, (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1)
	cv.circle(orig, (int(trbrX), int(trbrY)), 5, (0, 0, 255), -1)

	# Drawing the lines joining the midpoints of box on original image
	cv.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 255, 0), 2)
	cv.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 0), 2)

	# Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / 2.4

	# Compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	# Draw the object sizes on the image
	cv.putText(orig, "{:.1f}cm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
	cv.putText(orig, "{:.1f}cm".format(dimB), (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

	# Show the output image
	cv.imshow("Measure", orig)
	cv.waitKey(5000)

cv.destroyAllWindows()
