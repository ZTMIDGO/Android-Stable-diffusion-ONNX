package com.example.open.diffusion;

public abstract class MyRunnable implements Runnable {
    private boolean isCancel;

    public boolean isCancel() {
        return isCancel;
    }

    public void setCancel(boolean cancel) {
        isCancel = cancel;
    }
}
