from django import forms
from .models import Complaint


class complaintForm(forms.ModelForm):
    class Meta:
        model = Complaint
        fields = ['email','address','phone','social_media','messager_id','status','message']
        status=forms.ModelChoiceField
        social_media=forms.ModelChoiceField