"""Public facade for face-detail dialog components."""
from app_pages.face_dialogs import render_face_info_modal, render_name_assignment_modal
from app_pages.facial_analysis import show_analysis_modal, show_full_image_modal

__all__ = ["render_face_info_modal", "render_name_assignment_modal", "show_analysis_modal", "show_full_image_modal"]
