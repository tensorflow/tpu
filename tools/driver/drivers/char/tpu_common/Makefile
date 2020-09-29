#  Makefile for Tpu drivers.

subdir-ccflags-y 		:= -I$(srctree)

obj-y				+= tpuv1_nodriver.o
obj-$(CONFIG_TPU_COMMON_COMMON)    += tpu_common.o
obj-$(CONFIG_TPU_V1_CORE)    += tpu_v1_core.o
obj-$(CONFIG_TPU_V2_DRIVER)	+= tpu_v2.o
obj-$(CONFIG_TPU_V1_DRIVER)	+= tpu_v1.o

tpu_v1_core-objs		:= tpuv1_core.o
tpu_v1-objs		 	:= tpuv1_driver.o
tpu_v2-objs		 	:= tpuv2_driver.o
