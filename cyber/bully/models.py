from django.db import models


# Create your models here.
class user(models.Model):
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255)
    password = models.CharField(max_length=255, editable=False)
    rpwd = models.CharField(max_length=255, editable=False)

class feed(models.Model):
    name=models.CharField(max_length=255)
    email=models.EmailField(max_length=255)
    message=models.CharField(max_length=255)
    
class Image(models.Model):
    name = models.CharField(max_length=30)
    img = models.ImageField(upload_to='pix')


class Complaint(models.Model):
    STATUS_CHOICES = (
        ('OFFENSIVE', 'OFFENSIVE'),
        ('NOFFENSIVE', 'NOFFENSIVE')
    )
    social = (
        ('FACEBOOK', 'FACEBOOK'),
        ('INSTAGRAM', 'INSTAGRAM'),
        ('TWITTER', 'TWITTER')
    )
    complainer=models.CharField(max_length=255)
    email = models.EmailField()
    address = models.CharField(max_length=255)
    phone = models.PositiveIntegerField()
    status = models.CharField(choices=STATUS_CHOICES, max_length=10,null=True)
    social_media = models.CharField(choices=social, max_length=10,null=True)
    messager_id = models.CharField(max_length=255)
    message = models.FileField(upload_to='documents/')
