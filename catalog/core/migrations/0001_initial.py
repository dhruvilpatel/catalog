# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-04-27 00:08
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Creator',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('creator_type', models.CharField(choices=[(b'AUTHOR', 'author'), (b'REVIEWED_AUTHOR', 'reviewed author'), (b'CONTRIBUTOR', 'contributor'), (b'EDITOR', 'editor'), (b'TRANSLATOR', 'translator'), (b'SERIES_EDITOR', 'series editor')], max_length=32)),
                ('first_name', models.CharField(max_length=255)),
                ('last_name', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='InvitationEmailTemplate',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=32)),
                ('text', models.TextField()),
                ('date_added', models.DateTimeField(auto_now_add=True)),
                ('last_modified', models.DateTimeField(auto_now=True)),
                ('added_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Journal',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('url', models.URLField(blank=True, max_length=255)),
                ('abbreviation', models.CharField(blank=True, max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='ModelDocumentation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Note',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('date_added', models.DateTimeField(auto_now_add=True)),
                ('date_modified', models.DateTimeField(auto_now=True)),
                ('zotero_key', models.CharField(blank=True, max_length=64, null=True, unique=True)),
                ('zotero_date_added', models.DateTimeField(blank=True, null=True)),
                ('zotero_date_modified', models.DateTimeField(blank=True, null=True)),
                ('deleted_on', models.DateTimeField(blank=True, null=True)),
                ('added_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='added_note_set', to=settings.AUTH_USER_MODEL)),
                ('deleted_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='deleted_note_set', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Platform',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('url', models.URLField(blank=True, null=True)),
                ('description', models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='PlatformVersion',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('version', models.TextField()),
                ('platform', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.Platform')),
            ],
        ),
        migrations.CreateModel(
            name='Publication',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.TextField()),
                ('abstract', models.TextField(blank=True)),
                ('short_title', models.CharField(blank=True, max_length=255)),
                ('zotero_key', models.CharField(blank=True, max_length=64, null=True, unique=True)),
                ('url', models.URLField(blank=True)),
                ('date_published_text', models.CharField(blank=True, max_length=32)),
                ('date_published', models.DateField(blank=True, null=True)),
                ('date_accessed', models.DateField(blank=True, null=True)),
                ('archive', models.CharField(blank=True, max_length=255)),
                ('archive_location', models.CharField(blank=True, max_length=255)),
                ('library_catalog', models.CharField(blank=True, max_length=255)),
                ('call_number', models.CharField(blank=True, max_length=255)),
                ('rights', models.CharField(blank=True, max_length=255)),
                ('extra', models.TextField(blank=True)),
                ('published_language', models.CharField(blank=True, default=b'English', max_length=255)),
                ('zotero_date_added', models.DateTimeField(blank=True, help_text='date added field from zotero', null=True)),
                ('zotero_date_modified', models.DateTimeField(blank=True, help_text='date modified field from zotero', null=True)),
                ('code_archive_url', models.URLField(blank=True, max_length=255)),
                ('contact_author_name', models.CharField(blank=True, max_length=255)),
                ('contact_email', models.EmailField(blank=True, max_length=254)),
                ('status', models.CharField(choices=[(b'UNTAGGED', 'Not reviewed'), (b'NEEDS_AUTHOR_REVIEW', 'Curator has reviewed publication, requires author intervention.'), (b'FLAGGED', 'Flagged for further internal review by CoMSES staff'), (b'AUTHOR_UPDATED', 'Updated by author, needs CoMSES review'), (b'INVALID', 'Publication record is not applicable or invalid'), (b'COMPLETE', 'Reviewed and verified by CoMSES')], default=b'UNTAGGED', max_length=32)),
                ('date_added', models.DateTimeField(auto_now_add=True, help_text='Date this publication was imported into this system')),
                ('date_modified', models.DateTimeField(auto_now=True, help_text='Date this publication was last modified on this system')),
                ('author_comments', models.TextField(blank=True)),
                ('email_sent_count', models.PositiveIntegerField(default=0)),
                ('pages', models.CharField(blank=True, max_length=255, null=True)),
                ('issn', models.CharField(blank=True, max_length=255, null=True)),
                ('volume', models.CharField(blank=True, max_length=255, null=True)),
                ('issue', models.CharField(blank=True, max_length=255, null=True)),
                ('series', models.CharField(blank=True, max_length=255, null=True)),
                ('series_title', models.CharField(blank=True, max_length=255, null=True)),
                ('series_text', models.CharField(blank=True, max_length=255, null=True)),
                ('doi', models.CharField(blank=True, max_length=255, null=True)),
                ('added_by', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='added_publication_set', to=settings.AUTH_USER_MODEL)),
                ('assigned_curator', models.ForeignKey(blank=True, help_text='Currently assigned curator', null=True, on_delete=django.db.models.deletion.CASCADE, related_name='assigned_publication_set', to=settings.AUTH_USER_MODEL)),
                ('creators', models.ManyToManyField(to='core.Creator')),
                ('journal', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='core.Journal')),
                ('model_documentation', models.ManyToManyField(blank=True, to='core.ModelDocumentation')),
                ('platforms', models.ManyToManyField(blank=True, to='core.Platform')),
            ],
        ),
        migrations.CreateModel(
            name='PublicationAuditLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date_added', models.DateTimeField(auto_now_add=True)),
                ('action', models.CharField(choices=[(b'AUTHOR_EDIT', 'Author edit'), (b'SYSTEM_LOG', 'System log'), (b'CURATOR_EDIT', 'Curator edit')], default=b'SYSTEM_LOG', max_length=32)),
                ('message', models.TextField(blank=True)),
                ('creator', models.ForeignKey(blank=True, help_text='The user who initiated this action, if any.', null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('publication', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='audit_log_set', to='core.Publication')),
            ],
            options={
                'ordering': ['-date_added'],
            },
        ),
        migrations.CreateModel(
            name='Sponsor',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
                ('url', models.URLField(blank=True)),
                ('description', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255, unique=True)),
            ],
        ),
        migrations.AddField(
            model_name='publication',
            name='sponsors',
            field=models.ManyToManyField(blank=True, to='core.Sponsor'),
        ),
        migrations.AddField(
            model_name='publication',
            name='tags',
            field=models.ManyToManyField(blank=True, to='core.Tag'),
        ),
        migrations.AddField(
            model_name='note',
            name='publication',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='core.Publication'),
        ),
    ]
