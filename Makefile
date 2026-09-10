test:
	python recst.py
	python dtps.py
	python fgsm.py

clean:
	rm -rf losscurves results samples weights