<script setup lang="ts">
import Sidebar from './Sidebar.vue';
import Toolbar from './Toolbar.vue';
import StatusBar from './StatusBar.vue';
import StartupNotice from '../components/StartupNotice.vue';
import { onBeforeUnmount } from 'vue';
import { useAppStore } from '../stores/app';
const store = useAppStore();
store.initialize();
onBeforeUnmount(store.dispose);
</script>
<template>
  <div class="app-shell">
    <Sidebar />
    <div class="workspace">
      <Toolbar />
      <StartupNotice />
      <main class="workspace-content">
        <RouterView v-slot="{ Component, route }">
          <Transition name="page" mode="out-in">
            <component :is="Component" :key="route.path" />
          </Transition>
        </RouterView>
      </main>
      <StatusBar />
    </div>
  </div>
</template>
<style scoped>
.app-shell {
  height: 100%;
  width: 100%;
  display: flex;
}
.workspace {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}
.workspace-content {
  flex: 1;
  min-height: 0;
  position: relative;
  overflow: hidden;
}
.page-enter-active, .page-leave-active {
  transition: opacity var(--duration-page) var(--ease), transform var(--duration-page) var(--ease);
}
.page-enter-from {
  opacity: 0;
  transform: translateY(6px);
}
.page-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}
</style>
