'use client';
import { toast } from 'sonner';

interface ToastProps {
  title: string;
  description?: string;
}

export function toastAlert({ title, description }: ToastProps) {
  toast.warning(title, {
    description,
    position: 'top-left',
    duration: 10_000,
  });
}
