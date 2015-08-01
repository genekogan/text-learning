#!/usr/local/opt/python/bin/python2.7
"""
Print a description of a MIDI file.
"""

import midi
from midiutil.MidiFile import MIDIFile

midifile = "/Users/gene/Code/Python/python-midi/mary.mid"
pattern = midi.read_midifile(midifile)
#print repr(pattern)

def main_in():
	for track in pattern:
		print "track"
		for p in track:
			if p.name == "Note On":
				pitch, velocity, tick = [p.get_pitch(), p.get_velocity(), p.tick]
				print "note "+str(pitch)+", "+str(velocity)+", tick "+str(tick)

def main_out():	
	MyMIDI = MIDIFile(1)

	track = 0
	time = 0
	MyMIDI.addTrackName(track,time,"Sample Track")
	MyMIDI.addTempo(track,time, 120)

	# Add a note. addNote expects the following information:
	channel = 0
	pitch = 60
	duration = 1
	volume = 100

	# Now add the note.
	MyMIDI.addNote(track,channel,pitch,time,duration,volume)

	binfile = open("output.mid", 'wb')
	MyMIDI.writeFile(binfile)
	binfile.close()

	
#main_out()



