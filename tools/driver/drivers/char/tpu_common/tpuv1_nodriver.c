/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#include "tpuv1_nodriver.h"
#include "linux/delay.h"
#include "linux/jiffies.h"
#define TPUV1_TN_BAR 2
#define TPUV1_REG_CHIP_RESET 0xb000
#define TPUV1_REG_CHIP_INIT_DONE 0x470200
#define TPUV1_REG_MAX TPUV1_REG_CHIP_INIT_DONE
#define TPUV1_VALUE_CHIP_INIT_DONE 2
#define TPUV1_VALUE_CHIP_RESET 3
#define TPUV1_VALUE_RESET_ACCEPTED 0
#define TPUV1_RESET_TIMEOUT_SEC 5
static u8 __iomem *tpuv1_map_bar(struct pci_dev *pci_dev, int bar_num)
{
 ulong phys_addr, length_bytes;
 u8 __iomem *bar_mem;
 if (pci_enable_device(pci_dev)) {
  dev_err(&pci_dev->dev, "could not enable PCI dev");
  return NULL;
 }
 phys_addr = pci_resource_start(pci_dev, bar_num);
 if (!phys_addr) {
  dev_err(&pci_dev->dev, "Could not get start of BAR%d", bar_num);
  pci_disable_device(pci_dev);
  return NULL;
 }
 length_bytes = pci_resource_len(pci_dev, bar_num);
 if (length_bytes < TPUV1_REG_MAX) {
  dev_err(&pci_dev->dev, "Invalid length of BAR%d: %lu",
   bar_num, length_bytes);
  pci_disable_device(pci_dev);
  return NULL;
 }
 bar_mem = ioremap_nocache(phys_addr, length_bytes);
 if (!bar_mem) {
  pci_disable_device(pci_dev);
 }
 return bar_mem;
}
static void tpuv1_unmap_bar(
 struct pci_dev *pci_dev, int bar_num, u8 __iomem *bar_regs)
{
 iounmap(bar_regs);
 pci_disable_device(pci_dev);
}
int tpuv1_nodriver_reset(struct pci_dev *pci_dev)
{
 ulong value;
 ulong timeout;
 int ret = 0;
 u8 __iomem *bar2_regs;
 u16 pci_command;
 u16 *pci_command_reg;
 bar2_regs = tpuv1_map_bar(pci_dev, TPUV1_TN_BAR);
 if (!bar2_regs)
  return -ENOMEM;
 pci_read_config_word(pci_dev, PCI_COMMAND, &pci_command);
 pci_write_config_word(pci_dev, PCI_COMMAND,
    pci_command | PCI_COMMAND_MEMORY);
 pci_command_reg = (u16 *)((char *)pci_dev->saved_config_space +
    PCI_COMMAND);
 *pci_command_reg |= PCI_COMMAND_MEMORY;
 writeq(TPUV1_VALUE_CHIP_RESET, &bar2_regs[TPUV1_REG_CHIP_RESET]);
 timeout = jiffies + TPUV1_RESET_TIMEOUT_SEC * HZ;
 do {
  value = readq(&bar2_regs[TPUV1_REG_CHIP_RESET]);
  if (value == TPUV1_VALUE_RESET_ACCEPTED)
   break;
  udelay(1);
 } while (time_before(jiffies, timeout));
 if (value != TPUV1_VALUE_RESET_ACCEPTED) {
  dev_err(&pci_dev->dev,
   "(PCI hot reset): TPUV1 reset not accepted within timeout: 0x%lx.",
   value);
  ret = -ETIMEDOUT;
  goto out;
 }
 timeout = jiffies + TPUV1_RESET_TIMEOUT_SEC * HZ;
 do {
  value = readq(&bar2_regs[TPUV1_REG_CHIP_INIT_DONE]);
  if (value == TPUV1_VALUE_CHIP_INIT_DONE)
   break;
  udelay(1);
 } while (time_before(jiffies, timeout));
 if (value != TPUV1_VALUE_CHIP_INIT_DONE) {
  dev_err(&pci_dev->dev,
   "(PCI hot reset): TPUV1 reset not complete within timeout: 0x%lx",
   value);
  ret = -ETIMEDOUT;
 }
out:
 tpuv1_unmap_bar(pci_dev, TPUV1_TN_BAR, bar2_regs);
 return ret;
}
