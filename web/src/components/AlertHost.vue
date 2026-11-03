<script setup lang="ts">
import { activeConfirm, settleConfirm } from '../composables/confirm';
import UAlert from './ui/UAlert.vue';
</script>
<template>
  <UAlert :model-value="!!activeConfirm" :title="activeConfirm?.title ?? ''" :message="activeConfirm?.message ?? ''" :confirm-text="activeConfirm?.confirmText" :cancel-text="activeConfirm?.cancelText" :destructive="activeConfirm?.destructive" :alert-only="activeConfirm?.alertOnly" @confirm="settleConfirm(true)" @cancel="settleConfirm(false)" />
</template>
