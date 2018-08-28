import numpy
import cv2

def detectPatches(img, nodeId):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret,thresh = cv2.threshold(gray,127,255,1)

	_, contours, _ = cv2.findContours(thresh,1,2)

	patchIdx = 0

	for cnt in contours:
		approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

		if len(approx) == 4: # if rectangle
			x, y, width, height = cv2.boundingRect(cnt)
			roi = img[y: y + height, x: x + width]

			if(width > 10):
				avgColor = averageColorForImage(roi)
				imageFilename = "./output/node{}patch{}.png".format(nodeId, patchIdx)
				cv2.imwrite(imageFilename, roi)
				patchIdx = patchIdx + 1

def averageColorForImage(img):
	avg_color_per_row = numpy.average(img, axis=0)
	avg_color = numpy.average(avg_color_per_row, axis=0)
	return dict(
		r=avg_color[2], # values come not in order
		g=avg_color[1],
		b=avg_color[0]
	)
