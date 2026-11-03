<script setup lang="ts">
import { computed, ref, useId } from 'vue';
import { CircleHelp } from 'lucide-vue-next';
import UButton from './UButton.vue';
import { useOverlay } from '../../composables/overlays';
const props = withDefaults(defineProps<{
    modelValue: boolean;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    destructive?: boolean;
    alertOnly?: boolean;
}>(), { confirmText: '确认', cancelText: '取消', destructive: false });
const emit = defineEmits<{
    'update:modelValue': [
        value: boolean
    ];
    confirm: [
    ];
    cancel: [
    ];
}>();
const panel = ref<HTMLElement>();
const open = computed(() => props.modelValue);
const titleId = useId();
const messageId = useId();
function cancel() {
    if (props.alertOnly) {
        accept();
        return;
    }
    emit('cancel');
    emit('update:modelValue', false);
}
function accept() {
    emit('confirm');
    emit('update:modelValue', false);
}
useOverlay(open, panel, cancel, true);
</script>
<template>
  <Teleport to="body">
    <Transition name="mask">
      <div v-if="modelValue" class="overlay-mask alert-mask">
        <Transition name="alert" appear>
          <section v-if="modelValue" ref="panel" class="u-alert" role="alertdialog" aria-modal="true" :aria-labelledby="titleId" :aria-describedby="messageId" tabindex="-1">
            <div class="alert-icon">
              <CircleHelp :size="30" />
            </div>
            <h2 :id="titleId">{{ title }}</h2>
            <p :id="messageId" class="selectable">{{ message }}</p>
            <footer :class="{single: alertOnly}">
              <UButton v-if="!alertOnly" @click="cancel">{{ cancelText }}</UButton>
              <UButton :variant="destructive ? 'destructive' : 'primary'" @click="accept">{{ confirmText }}</UButton>
            </footer>
          </section>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>
<style scoped>
.alert-mask {
  align-items: center;
  z-index: var(--z-alert);
}
.u-alert {
  width: 340px;
  max-width: calc(100vw - 48px);
  border: 1px solid var(--separator);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  background: var(--surface);
  box-shadow: var(--shadow-modal);
  text-align: center;
  outline: none;
}
.alert-icon {
  display: grid;
  place-items: center;
  margin-bottom: var(--space-3);
  color: var(--accent);
}
h2 {
  font-size: var(--font-lg);
  font-weight: 600;
}
p {
  margin-top: var(--space-2);
  color: var(--secondary);
  font-size: var(--font-sm);
  line-height: 1.8;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
footer {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-2);
  margin-top: var(--space-6);
}
footer.single {
  grid-template-columns: 1fr;
}
.alert-enter-active, .alert-leave-active {
  transition: transform var(--duration-normal) var(--ease), opacity var(--duration-normal) var(--ease);
}
.alert-enter-from, .alert-leave-to {
  opacity: 0;
  transform: scale(.94);
}
.mask-leave-active .u-alert {
  transition: transform var(--duration-normal) var(--ease);
}
.mask-leave-to .u-alert {
  transform: scale(.94);
}
</style>
