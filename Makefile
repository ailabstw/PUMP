all:
	$(EXEC) mkdir build && cd build && cmake .. && make && cd .. && cp build/ulms.so .

clean:
	$(EXEC) rm -rf build/ ulms.so *~ utils/*~
