#  Makefile for source common to ASIC-SW drivers.

subdir-ccflags-y 		:= -I$(srctree)

obj-$(CONFIG_ASIC_SW)		+= asic_sw.o

asic_sw-objs			:= asic_sw_module.o asic_sw_firmware_indirect_registers.o \
				   asic_sw_clock.o
