.PHONY: all verify

all:
	./build_all.sh

verify:
	conda run -n ML python ../Source/reporting/verify_report_values.py --require-pdfs
