#  Makefile for tpu_common drivers.

subdir-ccflags-y 		:= -I$(srctree)

obj-y				+= nodriver.o
obj-y				+= tpuv2_nodriver.o
obj-y				+= tpuv4common_nodriver.o
obj-$(CONFIG_TPU_COMMON_COMMON)    += tpu_common.o
obj-$(CONFIG_TPU_V2_CORE)    += tpu_v2_core.o
obj-$(CONFIG_TPU_V3_DRIVER)	+= tpu_v3.o
obj-$(CONFIG_TPU_V2_DRIVER)	+= tpu_v2.o
obj-$(CONFIG_TPU_V4_COMMON_COMMON)	+= tpuv4common_common.o
obj-$(CONFIG_TPU_V4_DRIVER)	+= tpu_v4.o
obj-$(CONFIG_TPU_V4_LITE_DRIVER)	+= tpu_v4_lite.o

tpu_v2_core-objs		:= tpuv2_core.o
tpu_v2-objs		 	:= tpuv2_driver.o
tpu_v3-objs		 	:= tpuv3_driver.o
tpuv4common_common-objs			:= tpuv4common.o \
				   tpuv4common/tpuv4common_lst.o
tpu_v4-objs			:= tpuv4_driver.o \
				   tpuv4.o \
				   tpuv4/tpuv4_bar0_ranges.o \
				   tpuv4/tpuv4_bar2_ranges.o \
				   tpuv4/tpuv4_device_owner_offsets.o \
				   tpuv4/tpuv4_reinit_reset_offsets.o \
				   tpuv4/tpuv4_mgr_error_loperf_offsets.o \
				   tpuv4/tpuv4_firmware_state_offsets.o \
				   tpuv4/tpuv4_firmware_version_offsets.o \
				   tpuv4/tpuv4_userspace_firmware_indirect_accessor_offsets.o \
				   tpuv4/tpuv4_userspace_lst_port_indirect_offsets.o \
				   tpuv4/tpuv4_pcie_flr_status_offsets.o
tpu_v4_lite-objs			:= tpuv4lite_driver.o \
				   tpuv4lite/tpuv4lite_bar0_ranges.o \
				   tpuv4lite/tpuv4lite_bar2_ranges.o \
				   tpuv4lite/tpuv4lite_device_owner_offsets.o \
				   tpuv4lite/tpuv4lite_reinit_reset_offsets.o \
				   tpuv4lite/tpuv4lite_mgr_error_loperf_offsets.o \
				   tpuv4lite/tpuv4lite_firmware_state_offsets.o \
				   tpuv4lite/tpuv4lite_firmware_version_offsets.o \
				   tpuv4lite/tpuv4lite_userspace_firmware_indirect_accessor_offsets.o \
				   tpuv4lite/tpuv4lite_userspace_lst_port_indirect_offsets.o \
				   tpuv4lite/tpuv4lite_pcie_flr_status_offsets.o
