locations = "locationsro.txt"
kayitlar = "kayitall.txt"
output = "mixed.txt"

with open(locations, 'r') as f_loc:
    with open(output, 'w') as f_mix:
        for line in f_loc:
            parts = line.split('  ')
            timestamploc = parts[0]
            
            with open(kayitlar, 'r') as f_kay:
                rss1 = "_"
                rss2 = "_"
                rss3 = "_"
                for line in f_kay:
                    parts2 = line.split('  ')
                    timestamprss = parts2[0]
                    
                    if timestamploc == timestamprss:
                        if parts2[1] == "ExtremeNetwo_39:29:e8":
                            rss1 = parts2[3].split("\n")[0]
                        elif parts2[1] == "ExtremeNetwo_4a:06:c9":
                            rss2= parts2[3].split("\n")[0]
                        elif parts2[1] == "ExtremeNetwo_03:61:00":
                            rss3= parts2[3].split("\n")[0]
                    else:
                        continue
            mixed_data = timestamploc + " " + parts[1] + " " + parts[2].split("\n")[0] + " " + rss1 + " " + rss2 + " " + rss3 + " \n" 
            f_mix.write(mixed_data)

                
 

print("Timestamps rounded and written to output file.")