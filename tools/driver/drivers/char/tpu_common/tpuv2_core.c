/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/tpuv2_ioctl.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include "tpuv2_core.h"
#include "tpu_common.h"
#define TPUV2_CORE_VERSION "1.0.1"
static ssize_t sysfs_show(struct device *device, struct device_attribute *attr,
     char *buf);
static long tpuv2_set_tc_csr_access(struct gasket_dev *gasket_dev, ulong arg);
static inline int reset_complete(struct gasket_dev *gasket_dev,
     bool log_not_complete);
enum sysfs_attribute_type {
 ATTR_TN0_PAGE_TABLE_SIZE,
 ATTR_TN0_SIMPLE_PAGE_TABLE_SIZE,
 ATTR_TN0_NUM_ACTIVE_PAGES,
 ATTR_TN1_PAGE_TABLE_SIZE,
 ATTR_TN1_SIMPLE_PAGE_TABLE_SIZE,
 ATTR_TN1_NUM_ACTIVE_PAGES,
};
struct tpuv2_device_data {
 struct tpu_common_device_data tpu_common_data;
};
const struct gasket_bar_desc tpuv2_bar_descriptions[] = {
 { TPUV2_LBUS_BAR_BYTES, VM_READ, TPUV2_LBUS_BAR_OFFSET, TPUV2_NUM_LBUS_RANGES,
   lbus_mappable_regions },
 GASKET_UNUSED_BAR,
 { TPUV2_TN_BAR_BYTES, (VM_WRITE | VM_READ), TPUV2_TN_BAR_OFFSET,
   TPUV2_NUM_TN_RANGES, tn_mappable_regions },
 GASKET_UNUSED_BAR,
 GASKET_UNUSED_BAR,
 GASKET_UNUSED_BAR
};
struct gasket_page_table_config tpuv2_page_table_configs[] = {
 {
  .id = 0,
  .mode = GASKET_PAGE_TABLE_MODE_NORMAL,
  .total_entries = TPUV2_PAGE_TABLE_MAX,
  .bar_index = TPUV2_TN_BAR_INDEX,
  .base_reg = TPUV2_BAR2_REG_TN0_PAGE_TABLE,
  .extended_reg = TPUV2_BAR2_REG_TN0_EXTENDED_TABLE,
  .extended_bit = TPUV2_EXTENDED_SHIFT,
 },
 {
  .id = 1,
  .mode = GASKET_PAGE_TABLE_MODE_NORMAL,
  .total_entries = TPUV2_PAGE_TABLE_MAX,
  .bar_index = TPUV2_TN_BAR_INDEX,
  .base_reg = TPUV2_BAR2_REG_TN1_PAGE_TABLE,
  .extended_reg = TPUV2_BAR2_REG_TN1_EXTENDED_TABLE,
  .extended_bit = TPUV2_EXTENDED_SHIFT,
 },
};
EXPORT_SYMBOL(tpuv2_page_table_configs);
struct gasket_sysfs_attribute tpuv2_sysfs_attrs[] = {
 GASKET_SYSFS_RO(tensornode_0_page_table_entries, sysfs_show,
   ATTR_TN0_PAGE_TABLE_SIZE),
 GASKET_SYSFS_RO(tensornode_0_simple_page_table_entries, sysfs_show,
   ATTR_TN0_SIMPLE_PAGE_TABLE_SIZE),
 GASKET_SYSFS_RO(tensornode_0_num_mapped_pages, sysfs_show,
   ATTR_TN0_NUM_ACTIVE_PAGES),
 GASKET_SYSFS_RO(tensornode_1_page_table_entries, sysfs_show,
   ATTR_TN1_PAGE_TABLE_SIZE),
 GASKET_SYSFS_RO(tensornode_1_simple_page_table_entries, sysfs_show,
   ATTR_TN1_SIMPLE_PAGE_TABLE_SIZE),
 GASKET_SYSFS_RO(tensornode_1_num_mapped_pages, sysfs_show,
   ATTR_TN1_NUM_ACTIVE_PAGES),
 GASKET_SYSFS_REG(mgt_kernel_dft_kernel_tap,
    TPUV2_BAR2_REG_MGT_KERNEL_DFT_KERNEL_TAP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_dft_kernel_efuse_lsb,
    TPUV2_BAR2_REG_MGT_KERNEL_DFT_KERNEL_EFUSE_LSB,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_dft_kernel_efuse_msb,
    TPUV2_BAR2_REG_MGT_KERNEL_DFT_KERNEL_EFUSE_MSB,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_ltssm_state,
    TPUV2_BAR2_REG_LTSSM_STATE, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes0_debug_status,
    TPUV2_BAR2_REG_SERDES0_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes1_debug_status,
    TPUV2_BAR2_REG_SERDES1_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes2_debug_status,
    TPUV2_BAR2_REG_SERDES2_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes3_debug_status,
    TPUV2_BAR2_REG_SERDES3_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes4_debug_status,
    TPUV2_BAR2_REG_SERDES4_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes5_debug_status,
    TPUV2_BAR2_REG_SERDES5_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes6_debug_status,
    TPUV2_BAR2_REG_SERDES6_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes7_debug_status,
    TPUV2_BAR2_REG_SERDES7_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes8_debug_status,
    TPUV2_BAR2_REG_SERDES8_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes9_debug_status,
    TPUV2_BAR2_REG_SERDES9_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes10_debug_status,
    TPUV2_BAR2_REG_SERDES10_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes11_debug_status,
    TPUV2_BAR2_REG_SERDES11_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes12_debug_status,
    TPUV2_BAR2_REG_SERDES12_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes13_debug_status,
    TPUV2_BAR2_REG_SERDES13_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes14_debug_status,
    TPUV2_BAR2_REG_SERDES14_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_ctrl_clk_pcie_serdes15_debug_status,
    TPUV2_BAR2_REG_SERDES15_DEBUG_STATUS, TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_0_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_0_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_0_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_0_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_0_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_0_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_1_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_1_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_1_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_1_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_nm_1_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_NM_1_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_0_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_0_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_0_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_0_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_0_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_0_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_1_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_1_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_1_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_1_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_1_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_1_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_2_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_2_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_2_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_2_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_2_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_2_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_3_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_3_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_3_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_3_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_3_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_3_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_4_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_4_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_4_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_4_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_4_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_4_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_0_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_0_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_1_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_1_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_2_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_2_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_3_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_3_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pll_4_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PLL_4_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_pll_pll_dll_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PCIE_PLL_PLL_DLL_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_pll_fatal_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PCIE_PLL_FATAL_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_pll_first_error_status,
    TPUV2_BAR2_REG_MGT_KERNEL_PCIE_PLL_FIRST_ERROR_STATUS,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_pcie_pll_first_error_timestamp,
    TPUV2_BAR2_REG_MGT_KERNEL_PCIE_PLL_FIRST_ERROR_TIMESTAMP,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_to_mirror_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_TO_MIRROR_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_0_mirrored_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_0_MIRRORED_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_to_mirror_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_TO_MIRROR_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_1_mirrored_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_1_MIRRORED_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_to_mirror_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_TO_MIRROR_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_2_mirrored_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_2_MIRRORED_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_to_mirror_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_TO_MIRROR_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_0,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_0,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_1,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_1,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_2,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_2,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_3,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_3,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_4,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_4,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_5,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_5,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_6,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_6,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_7,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_7,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_8,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_8,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_9,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_9,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_10,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_10,
    TPUV2_TN_BAR_INDEX),
 GASKET_SYSFS_REG(mgt_kernel_lst_3_mirrored_11,
    TPUV2_BAR2_REG_MGT_KERNEL_LST_3_MIRRORED_11,
    TPUV2_TN_BAR_INDEX),
 GASKET_END_OF_ATTR_ARRAY
};
const struct gasket_mappable_region tn_mappable_regions[TPUV2_NUM_TN_RANGES] = {
 { 0x400000, 0x400000, VM_READ | VM_WRITE },
 { 0xa00000, 0xe00000, VM_READ | VM_WRITE },
 { 0x1a00000, 0xe00000, VM_READ | VM_WRITE },
};
EXPORT_SYMBOL(tn_mappable_regions);
const struct gasket_mappable_region
  lbus_mappable_regions[TPUV2_NUM_LBUS_RANGES] = {
 { 0x0, 0x1000000, VM_READ }
};
EXPORT_SYMBOL(lbus_mappable_regions);
int tpuv2_add_dev_cb(struct gasket_dev *gasket_dev)
{
 int ret = 0;
 struct tpuv2_device_data *tpuv2_device =
  kzalloc(sizeof(struct tpuv2_device_data), GFP_KERNEL);
 if (!tpuv2_device) {
  gasket_log_error(
   gasket_dev,
   "Unable to initialize tpu_v2 device storage.");
  return -ENOMEM;
 }
 ret = tpu_common_setup_device_data(
  &(tpuv2_device->tpu_common_data), TPUV2_CHIP_REINIT_RESET,
                       TPUV2_TN_BAR_INDEX,
  TPUV2_BAR2_REG_MGT_KERNEL_IS_DEVICE_OWNED_REGISTER,
                              0);
 if (ret)
  goto setup_failed;
 gasket_dev->cb_data = tpuv2_device;
 return 0;
setup_failed:
 kfree(tpuv2_device);
 return ret;
}
EXPORT_SYMBOL(tpuv2_add_dev_cb);
int tpuv2_remove_dev_cb(struct gasket_dev *gasket_dev)
{
 if (gasket_dev->cb_data == NULL)
  return -EINVAL;
 kfree(gasket_dev->cb_data);
 gasket_dev->cb_data = NULL;
 return 0;
}
EXPORT_SYMBOL(tpuv2_remove_dev_cb);
int tpuv2_sysfs_setup_cb(struct gasket_dev *gasket_dev)
{
 int ret;
 ret = tpu_common_sysfs_setup(gasket_dev);
 if (ret)
  return ret;
 return gasket_sysfs_create_entries(&gasket_dev->accel_dev.dev,
        tpuv2_sysfs_attrs);
}
EXPORT_SYMBOL(tpuv2_sysfs_setup_cb);
int tpuv2_device_open_cb(struct gasket_filp_data *filp_data, struct file *file)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv2_device_data *device_data = gasket_dev->cb_data;
 return tpu_common_device_open(gasket_dev, &(device_data->tpu_common_data),
                                 1);
}
EXPORT_SYMBOL(tpuv2_device_open_cb);
enum gasket_status tpuv2_get_status(struct gasket_dev *gasket_dev)
{
 ulong init_val;
 ulong error_status;
 init_val = gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
          TPUV2_BAR2_REG_MGT_CHIP_INIT_DONE);
 if (init_val != TPUV2_CHIP_INIT_DONE) {
  gasket_log_error(gasket_dev, "Chip init register value: %lu",
     init_val);
  return GASKET_STATUS_DEAD;
 }
 error_status =
  gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
       TPUV2_BAR2_REG_MGT_GLOBAL_FATAL_ERROR_STATUS);
 if (error_status) {
  gasket_log_error(gasket_dev, "Global error status is 0x%lx",
     error_status);
  return GASKET_STATUS_DEAD;
 }
 return GASKET_STATUS_ALIVE;
}
EXPORT_SYMBOL(tpuv2_get_status);
int tpuv2_device_cleanup(struct gasket_filp_data *filp_data, struct file *file)
{
 ulong error_status;
 uint retry;
 ulong tn0_paused;
 ulong tn1_paused;
 uint i;
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 struct tpuv2_device_data *device_data = gasket_dev->cb_data;
 error_status =
  gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
       TPUV2_BAR2_REG_MGT_GLOBAL_FATAL_ERROR_STATUS);
 if (error_status) {
  gasket_dev_write_64(gasket_dev, 1, TPUV2_TN_BAR_INDEX,
        TPUV2_BAR2_REG_TN0_DMA_PAUSE);
  gasket_dev_write_64(gasket_dev, 1, TPUV2_TN_BAR_INDEX,
        TPUV2_BAR2_REG_TN1_DMA_PAUSE);
  for (retry = 0; retry < TPUV2_RESET_RETRY; retry++) {
   tn0_paused =
    gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
         TPUV2_BAR2_REG_TN0_DMA_PAUSED);
   tn1_paused =
    gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
         TPUV2_BAR2_REG_TN1_DMA_PAUSED);
   if (tn0_paused && tn1_paused)
    break;
   set_current_state(TASK_UNINTERRUPTIBLE);
   schedule_timeout(msecs_to_jiffies(TPUV2_RESET_DELAY));
  }
  if ((retry > TPUV2_RESET_RETRY) && (!tn0_paused || !tn1_paused))
   gasket_log_error(gasket_dev, "dma pause timed out.");
  for (i = 0; i < TPUV2_NUM_TENSOR_NODES; ++i)
   gasket_page_table_unmap_all(gasket_dev->page_table[i]);
  gasket_dev->status = GASKET_STATUS_DEAD;
  gasket_log_error(gasket_dev, "non-zero error_status: 0x%lx",
     error_status);
 } else {
  if (device_data->tpu_common_data.reset_on_close)
   gasket_reset_nolock(gasket_dev, TPUV2_CHIP_REINIT_RESET);
 }
 tpu_common_clear_fw_device_owned(gasket_dev, &(device_data->tpu_common_data));
 return 0;
}
EXPORT_SYMBOL(tpuv2_device_cleanup);
int tpuv2_reset(struct gasket_dev *gasket_dev, uint type)
{
 if (type != TPUV2_CHIP_REINIT_RESET) {
  gasket_log_error(gasket_dev, "invalid reset type specified: %u",
     type);
  return -EINVAL;
 }
 return tpu_common_reinit_reset(gasket_dev, TPUV2_TN_BAR_INDEX, TPUV2_RESET_RETRY,
        TPUV2_RESET_DELAY, reset_complete,
        TPUV2_BAR2_REG_MGT_CHIP_RESET_REGISTER,
        TPUV2_CHIP_REINIT_RESET, TPUV2_RESET_ACCEPTED);
}
EXPORT_SYMBOL(tpuv2_reset);
int tpuv2_get_mappable_regions_cb(
 struct gasket_filp_data *filp_data, int bar_index,
 struct gasket_mappable_region **mappable_regions,
 int *num_mappable_regions)
{
 struct gasket_dev *gasket_dev = filp_data->gasket_dev;
 const struct gasket_mappable_region *source_regions;
 int i;
 if (bar_index == TPUV2_LBUS_BAR_INDEX) {
  *num_mappable_regions = TPUV2_NUM_LBUS_RANGES;
  source_regions = lbus_mappable_regions;
 } else if (bar_index == TPUV2_TN_BAR_INDEX) {
  *num_mappable_regions = TPUV2_NUM_TN_RANGES;
  source_regions = tn_mappable_regions;
 } else {
  gasket_log_error(gasket_dev, "Invalid BAR specified: %d",
     bar_index);
  return -EINVAL;
 }
 *mappable_regions = kzalloc(sizeof(struct gasket_mappable_region) *
         *num_mappable_regions,
        GFP_KERNEL);
 if (*mappable_regions == NULL) {
  gasket_log_error(gasket_dev,
     "Unable to allocate mappable regions!");
  return -ENOMEM;
 }
 for (i = 0; i < *num_mappable_regions; i++) {
  (*mappable_regions)[i].start = source_regions[i].start;
  (*mappable_regions)[i].length_bytes =
   source_regions[i].length_bytes;
  (*mappable_regions)[i].flags = source_regions[i].flags;
 }
 return 0;
}
EXPORT_SYMBOL(tpuv2_get_mappable_regions_cb);
static uint tpuv2_ioctl_check_cmd(uint cmd)
{
 switch (cmd) {
 case TPUV2_IOCTL_SET_DEBUG_TC_CSR_ACCESS:
  return 1;
 default:
  return 0;
 }
}
static uint tpuv2_ioctl_check_permissions(
 struct gasket_dev *gasket_dev, struct file *filp, uint cmd)
{
 int root = capable(CAP_SYS_ADMIN);
 fmode_t write = filp->f_mode & FMODE_WRITE;
 int device_owner = (gasket_dev->ownership.is_owned &&
       current->tgid == gasket_dev->ownership.owner);
 switch (cmd) {
 case TPUV2_IOCTL_SET_DEBUG_TC_CSR_ACCESS:
  return root || (write && device_owner);
 }
 return 0;
}
long tpuv2_ioctl(struct file *filp, uint cmd, ulong arg)
{
 struct gasket_filp_data *filp_data =
  (struct gasket_filp_data *)filp->private_data;
 struct gasket_dev *dev = filp_data->gasket_dev;
 if (!tpuv2_ioctl_check_cmd(cmd))
  return -ENOTTY;
 if (!tpuv2_ioctl_check_permissions(dev, filp, cmd))
  return -EPERM;
 switch (cmd) {
 case TPUV2_IOCTL_SET_DEBUG_TC_CSR_ACCESS:
  return tpuv2_set_tc_csr_access(dev, arg);
 }
 return -ENOTTY;
}
EXPORT_SYMBOL(tpuv2_ioctl);
static long tpuv2_set_tc_csr_access(struct gasket_dev *gasket_dev, ulong arg)
{
 struct tpuv2_tc_csr_access tpuv2_tc_csr_access;
 if (copy_from_user(&tpuv2_tc_csr_access, (void __user *)arg,
      sizeof(tpuv2_tc_csr_access)))
  return -EFAULT;
 if (tpuv2_tc_csr_access.enable != 0 && tpuv2_tc_csr_access.enable != 1) {
  gasket_log_error(gasket_dev,
     "Invalid value for enable flag: %d",
     tpuv2_tc_csr_access.enable);
  return -EINVAL;
 }
 if (tpuv2_tc_csr_access.tensor_node == 0) {
  gasket_dev_write_64(gasket_dev, tpuv2_tc_csr_access.enable,
        TPUV2_TN_BAR_INDEX,
        TPUV2_BAR2_REG_TN0_DEBUG_TC_CSR_ACCESS);
 } else if (tpuv2_tc_csr_access.tensor_node == 1) {
  gasket_dev_write_64(gasket_dev, tpuv2_tc_csr_access.enable,
        TPUV2_TN_BAR_INDEX,
        TPUV2_BAR2_REG_TN1_DEBUG_TC_CSR_ACCESS);
 } else {
  gasket_log_error(gasket_dev,
     "Invalid TensorNode index specified: %d",
     tpuv2_tc_csr_access.tensor_node);
  return -EINVAL;
 }
 return 0;
}
static ssize_t sysfs_show(struct device *device, struct device_attribute *attr,
     char *buf)
{
 int ret;
 struct gasket_dev *gasket_dev;
 struct gasket_sysfs_attribute *gasket_attr;
 struct tpuv2_device_data *tpuv2_device;
 enum sysfs_attribute_type type;
 gasket_dev = gasket_sysfs_get_device_data(device);
 if (gasket_dev == NULL)
  return 0;
 tpuv2_device = gasket_dev->cb_data;
 gasket_attr = gasket_sysfs_get_attr(device, attr);
 if (gasket_attr == NULL) {
  return 0;
 }
 type = (enum sysfs_attribute_type)gasket_attr->data.attr_type;
 switch (type) {
 case ATTR_TN0_PAGE_TABLE_SIZE:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_entries(
     gasket_dev->page_table[0]));
  break;
 case ATTR_TN0_SIMPLE_PAGE_TABLE_SIZE:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_entries(
     gasket_dev->page_table[0]));
  break;
 case ATTR_TN0_NUM_ACTIVE_PAGES:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_active_pages(
     gasket_dev->page_table[0]));
  break;
 case ATTR_TN1_PAGE_TABLE_SIZE:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_entries(
     gasket_dev->page_table[1]));
  break;
 case ATTR_TN1_SIMPLE_PAGE_TABLE_SIZE:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_entries(
     gasket_dev->page_table[1]));
  break;
 case ATTR_TN1_NUM_ACTIVE_PAGES:
  ret = scnprintf(buf, PAGE_SIZE, "%u\n",
    gasket_page_table_num_active_pages(
     gasket_dev->page_table[1]));
  break;
 default:
  gasket_log_error(gasket_dev, "Unknown attribute: %s",
     attr->attr.name);
  ret = 0;
  break;
 }
 return ret;
}
static inline int reset_complete(struct gasket_dev *gasket_dev,
     bool log_not_complete)
{
 int ret = 0;
 ulong init_val;
 ulong reset_val;
 init_val = gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
          TPUV2_BAR2_REG_MGT_CHIP_INIT_DONE);
 reset_val = gasket_dev_read_64(gasket_dev, TPUV2_TN_BAR_INDEX,
           TPUV2_BAR2_REG_MGT_CHIP_RESET_REGISTER);
 if ((init_val != TPUV2_CHIP_INIT_DONE) ||
     (reset_val != TPUV2_RESET_ACCEPTED)) {
  ret = -EBUSY;
 }
 if (log_not_complete && ret == -EBUSY) {
  gasket_log_error(
   gasket_dev,
   "Device is currently busy. Firmware state value: %lu; reset register value %lu",
   init_val, reset_val);
 }
 return ret;
}
MODULE_DESCRIPTION("Google tpu_v2 Driver Core");
MODULE_VERSION(TPUV2_CORE_VERSION);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Googler <noreply@google.com>");
