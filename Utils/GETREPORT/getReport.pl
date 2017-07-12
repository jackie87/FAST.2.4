#!/usr/bin/perl -w

use strict;
use lib '.';
use Statistics::Distributions;

if(@ARGV !=4){
  print "syntax:\n./getReport <file name> <model> <pval cut for gene> <pval cut for snp>\n";
  print "<file name> and <model>: getReport will look for files such as <file name>.minSNP.<model>.txt...\n";
  exit;
}
my $loc=shift;
my $model = shift;
my $gene_cut = shift;
my $SNP_cut = shift;

if( ! defined $model || ( $model ne "Linear" && $model ne "Logistic" && $model ne "Summary")){
  print "Type has to be specified, choose from Linear / Logistic / Summary\n";
  exit;
}

my $SNP_chi2_cut;
 if($SNP_cut>0) { 
    $SNP_chi2_cut = Statistics::Distributions::chisqrdistr(1, $SNP_cut);
 }else
 {
    $SNP_chi2_cut = 100000;
 }

sub read_GWiS_file{
  #for *.GWiS.$model.txt
  my $file = shift;
  my $gene = shift;
  my $ccds = shift;
  my $noPerm = 0;

  open(IN, $file);
  while(<IN>){
    if(/^Chr/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    if(/SUMMARY/){
      $gene->{$tok[1]}{chr} = $tok[0];
      $gene->{$tok[1]}{name} = $tok[2];
      $gene->{$tok[1]}{k} = $tok[12];
      if($model eq "Linear"){
	$gene->{$tok[1]}{chi2} = $tok[15];
	$gene->{$tok[1]}{bic} = $tok[14];
	$gene->{$tok[1]}{pval} = $tok[19];
	if($tok[19] eq "-"){
	  $noPerm=1;
	}
      }elsif($model eq "Logistic"){
	$gene->{$tok[1]}{chi2} = $tok[14];
	$gene->{$tok[1]}{bic} = $tok[13];
	$gene->{$tok[1]}{pval} = $tok[17];
	if($tok[17] eq "-"){
	  $noPerm=1;
	}
      }elsif($model eq "Summary"){
	$gene->{$tok[1]}{chi2} = $tok[15];
	$gene->{$tok[1]}{bic} = $tok[14];
	$gene->{$tok[1]}{pval} = $tok[19];
	if($tok[19] eq "-"){
	  $noPerm=1;
	}
      }

      $gene->{$tok[1]}{dat} = $_;

      $ccds->{$tok[1]}{chr} = $tok[0];
      $ccds->{$tok[1]}{name} = $tok[2];
      $ccds->{$tok[1]}{start} = $tok[3];
      $ccds->{$tok[1]}{end} = $tok[4];
      $ccds->{$tok[1]}{n} = $tok[6];
      $ccds->{$tok[1]}{v} = $tok[7];
      

    }else{
      
      if($tok[8] eq "NONE"){
	@{$gene->{$tok[1]}{lines}}=();
	@{$gene->{$tok[1]}{snps}}=();
      }
      
      push @{$gene->{$tok[1]}{lines}}, $_;
      if($tok[8] ne "NONE" &&$tok[8] ne "-"  ){
	push @{$gene->{$tok[1]}{snps}}, $tok[8];
	push @{$gene->{$tok[1]}{snp_pos}}, $tok[9];
#	push @{$gene->{$tok[1]}{snp_ssm}}, $tok[13]; #not applicable for logistic model
#	push @{$gene->{$tok[1]}{snp_chi2}}, $tok[15];#not applicable for logistic model
      }
    }
  }
  close(IN);
  return 1-$noPerm;
}

sub read_minSNP_file{
  #for *.minSNP.$model.txt or *.minSNP_Gene.$model.txt
  
  my $file = shift;
  my $gene = shift;
  my $ccds = shift;
  my $noPerm =0;

  open(IN, $file);
  while(<IN>){
    if(/^Chr/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    $gene->{$tok[1]}{chr} = $tok[0];
    $gene->{$tok[1]}{name} = $tok[2];

    $ccds->{$tok[1]}{start} = $tok[3];
    $ccds->{$tok[1]}{end} = $tok[4];
    $ccds->{$tok[1]}{chr} = $tok[0];
    $ccds->{$tok[1]}{n} = $tok[6];
    $ccds->{$tok[1]}{v} = $tok[7];

    if(exists($gene->{$tok[1]}{snp})){
      if($tok[15] <  $gene->{$tok[1]}{pval}|| ($tok[15] == 0 &&   $tok[12] >$gene->{$tok[1]}{chi2}) ) {
	$gene->{$tok[1]}{chi2} = $tok[12];
	$gene->{$tok[1]}{pval} = $tok[15];
	$gene->{$tok[1]}{snp} = $tok[8];
      }
    }else{
      $gene->{$tok[1]}{chi2} = $tok[12];
      $gene->{$tok[1]}{pval} = $tok[15];
      $gene->{$tok[1]}{snp} = $tok[8];
    }

    if($tok[15] eq "-"){
      $noPerm=1;
    }

  }
  close(IN);
  return 1-$noPerm;
}

sub read_BimBam_file{
  #for *.BF.$model.txt
  
  my $file = shift;
  my $gene = shift;
  my $ccds = shift;
  my $noPerm = 0;

  open(IN, $file);
  while(<IN>){
    if(/^Chr/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    $gene->{$tok[1]}{chr} = $tok[0];
    $gene->{$tok[1]}{name} = $tok[2];
    $gene->{$tok[1]}{pval} = $tok[11];
    $gene->{$tok[1]}{BF_sum} = $tok[8];

    $ccds->{$tok[1]}{start} = $tok[3];
    $ccds->{$tok[1]}{end} = $tok[4];
    $ccds->{$tok[1]}{chr} = $tok[0];
    $ccds->{$tok[1]}{n} = $tok[6];
    $ccds->{$tok[1]}{v} = $tok[7];

    if($tok[11] eq "-"){
      $noPerm=1;
    }

  }
  close(IN);
  return 1-$noPerm;
}


sub read_VEGAS_file{
  #for *.Vegas.$model.txt
  
  my $file = shift;
  my $gene = shift;
  my $ccds = shift;
  my $noPerm = 0;
  open(IN, $file);
  while(<IN>){
    if(/^Chr/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    $gene->{$tok[1]}{chr} = $tok[0];
    $gene->{$tok[1]}{name} = $tok[2];
    $gene->{$tok[1]}{pval} = $tok[11];
    $gene->{$tok[1]}{VEGAS_sum} = $tok[8];

    $ccds->{$tok[1]}{start} = $tok[3];
    $ccds->{$tok[1]}{end} = $tok[4];
    $ccds->{$tok[1]}{chr} = $tok[0];
    $ccds->{$tok[1]}{n} = $tok[6];
    $ccds->{$tok[1]}{v} = $tok[7];

    if($tok[11] eq "-"){
      $noPerm=1;
    }

  }
  close(IN);
  return 1-$noPerm;
}


sub read_Gates_file{
  #for *.Gates.$model.txt
  
  my $file = shift;
  my $gene = shift;
  my $ccds = shift;
  my $noPerm = 0;
  open(IN, $file);
  while(<IN>){
    if(/^Chr/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    $gene->{$tok[1]}{chr} = $tok[0];
    $gene->{$tok[1]}{name} = $tok[2];
    $gene->{$tok[1]}{pval} = $tok[9];
    $gene->{$tok[1]}{Gates} = $tok[8];

    $ccds->{$tok[1]}{start} = $tok[3];
    $ccds->{$tok[1]}{end} = $tok[4];
    $ccds->{$tok[1]}{chr} = $tok[0];
    $ccds->{$tok[1]}{n} = $tok[6];
    $ccds->{$tok[1]}{v} = $tok[7];

    if($tok[9] eq "-"){
      $noPerm=1;
    }

  }
  close(IN);
  return 1-$noPerm;
}

sub read_snp_result_file{
  #for *.allSNP.$model.txt
  my $file = shift;
  my $snp = shift;

  open(IN, $file);
  while(<IN>){
    if(/^SNP.id/){
      next;
    }
    chomp;
    my @tok = split/\t/;
    $snp->{$tok[0]}{chr} = $tok[1];
    $snp->{$tok[0]}{pos} = $tok[2];
    $snp->{$tok[0]}{chi2} = $tok[7];
    $snp->{$tok[0]}{MAF} = $tok[9];
    $snp->{$tok[0]}{R2} = $tok[10];
    $snp->{$tok[0]}{eSample} = $tok[11];
    $snp->{$tok[0]}{nGene} = $tok[12];
  }
  close(IN);
}


sub SNPdist2Gene{
  
  my $chr = shift;
  my $pos = shift;
  my $ccds = shift;
  
  my $dist=999999999;
  my $dist_mid=999999999;
  my $gene;
  
  for my $g ( sort {$ccds->{$a}{chr} <=> $ccds->{$b}{chr} || $ccds->{$a}{start} <=> $ccds->{$b}{start} } keys %{$ccds}){
    
    if($chr < $ccds->{$g}{chr}){
      next;
    }elsif($chr < $ccds->{$g}{chr}){
      last;
    }
    
    if($pos > $ccds->{$g}{start} && $pos < $ccds->{$g}{end}){
      $gene = $g;
      $dist = 0;
      $dist_mid = abs(($ccds->{$g}{start}+$ccds->{$g}{end})/2 - $pos);
      last;
    }
    
    if(abs($ccds->{$g}{start} -$pos) < $dist ){
      $gene = $g;
      $dist = abs($ccds->{$g}{start} -$pos);
      $dist_mid=abs(($ccds->{$g}{start}+$ccds->{$g}{end})/2 - $pos);
    }
    if(abs($ccds->{$g}{end} -$pos) < $dist  ){
      $gene = $g;
      $dist = abs($ccds->{$g}{end} -$pos);
      $dist_mid = abs(($ccds->{$g}{start}+$ccds->{$g}{end})/2 -$pos);
    }
  }
  return ($gene, $dist, $dist_mid);
}

my %GWiS;
my %minSNP;
my %minSNPP;
my %BimBam;
my %Vegas;
my %Gates;
my %SNP;
my %GENE;

my $flag_gene_result_file =0;
my $flag_snp_result_file =0;
if (-e "$loc.GWiS.$model.txt"){
  print "Found GWiS result file: $loc.GWiS.$model.txt\n";
  if (read_GWiS_file("$loc.GWiS.$model.txt", \%GWiS, \%GENE) == 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %GWiS);
    foreach my $g (sort {$GWiS{$b}{bic} <=> $GWiS{$a}{bic}} keys %GWiS){
      $GWiS{$g}{pval} = $counter/$total;
      $counter++;
    }
  }
  $flag_gene_result_file =1;
}
if (-e "$loc.minSNP.$model.txt"){
  print "Found minSNP result file: $loc.minSNP.$model.txt\n";
  if (read_minSNP_file("$loc.minSNP.$model.txt", \%minSNP, \%GENE) == 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %minSNP);
    foreach my $g (sort {$minSNP{$b}{chi2} <=> $minSNP{$a}{chi2}} keys %minSNP){
      $minSNP{$g}{pval} = $counter/$total;
      $counter++;
    }
  }
  $flag_gene_result_file =1;
}
if (-e "$loc.minSNP_Gene.$model.txt"){
  print "Found minSNP-P result file: $loc.minSNP_Gene.$model.txt\n";
  if (read_minSNP_file("$loc.minSNP_Gene.$model.txt", \%minSNPP, \%GENE) == 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %minSNPP);
    foreach my $g (sort {$minSNPP{$b}{chi2} <=> $minSNPP{$a}{chi2}} keys %minSNPP){
      $minSNPP{$g}{pval} = $counter/$total;
      $counter++;
    }
  }
  $flag_gene_result_file =1;
}
if (-e "$loc.BF.$model.txt"){
  print "Found BimBam result file: $loc.BF.$model.txt\n";
  if (read_BimBam_file("$loc.BF.$model.txt", \%BimBam, \%GENE) == 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %BimBam);
    foreach my $g (sort {$BimBam{$b}{BF_sum} <=> $BimBam{$a}{BF_sum}} keys %BimBam){
      $BimBam{$g}{pval} = $counter/$total;
      $counter++;
    }
  }

  $flag_gene_result_file =1;
}
if (-e "$loc.Vegas.$model.txt"){
  print "Found VEGAS result file: $loc.Vegas.$model.txt\n";
  if ( read_VEGAS_file("$loc.Vegas.$model.txt", \%Vegas, \%GENE)== 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %Vegas);
    foreach my $g (sort {$Vegas{$b}{VEGAS_sum} <=> $Vegas{$a}{VEGAS_sum}} keys %Vegas){
      $Vegas{$g}{pval} = $counter/$total;
      $counter++;
    }
  }

  $flag_gene_result_file =1;
}
if (-e "$loc.Gates.$model.txt"){
  print "Found Gates result file: $loc.Gates.$model.txt\n";
  if ( read_Gates_file("$loc.Gates.$model.txt", \%Gates, \%GENE)== 0){
    print "-No permutations have been performed, use rank/total_number_of_genes as p value\n";
    my $counter=1;
    my $total = scalar(keys %Gates);
    foreach my $g (sort {$Gates{$b}{VEGAS_sum} <=> $Gates{$a}{VEGAS_sum}} keys %Gates){
      $Gates{$g}{pval} = $counter/$total;
      $counter++;
    }
  }

  $flag_gene_result_file =1;
}
if (-e "$loc.allSNP.$model.txt"){
  print "Found all SNP result file: $loc.allSNP.$model.txt\n";
  if($flag_gene_result_file == 0){
    print "Need result file for at least one gene-based test\n";
    exit;
  }
  read_snp_result_file("$loc.allSNP.$model.txt", \%SNP);
  $flag_snp_result_file =1;
}

if( $flag_snp_result_file==0 && $flag_gene_result_file==0){
  print "No input files for get report present\n";
  exit;
}

my %GWS_genes;

foreach my $g (sort {$GWiS{$a}{pval} <=> $GWiS{$b}{pval}} keys %GWiS){
  if($GWiS{$g}{pval} < $gene_cut){
    $GWS_genes{$g} = 1;
  }
}

foreach my $g (sort {$minSNP{$a}{pval} <=> $minSNP{$b}{pval}} keys %minSNP){
  #if($minSNP{$g}{pval} < $SNP_cut && $minSNP{$g}{chi2}>$SNP_chi2_cut ){
  if($minSNP{$g}{pval} < $gene_cut ){
    $GWS_genes{$g} = 1;
  }
}

foreach my $g (sort {$minSNPP{$a}{pval} <=> $minSNPP{$b}{pval}} keys %minSNPP){
  if($minSNPP{$g}{pval} < $gene_cut){
    $GWS_genes{$g} = 1;
  }
}
foreach my $g (sort {$BimBam{$a}{pval} <=> $BimBam{$b}{pval}} keys %BimBam){
  if($BimBam{$g}{pval} < $gene_cut){
    $GWS_genes{$g} = 1;
  }
}
foreach my $g (sort {$Vegas{$a}{pval} <=> $Vegas{$b}{pval}} keys %Vegas){
  if($Vegas{$g}{pval} < $gene_cut){
    $GWS_genes{$g} = 1;
  }
}
foreach my $g (sort {$Gates{$a}{pval} <=> $Gates{$b}{pval}} keys %Gates){
  if($Gates{$g}{pval} < $gene_cut){
    $GWS_genes{$g} = 1;
  }
}
my %GWS_SNP=();
foreach my $snp (sort {$SNP{$b}{chi2} <=> $SNP{$a}{chi2}} grep { $SNP{$_}{chi2} == $SNP{$_}{chi2} } keys %SNP){
  #print $snp . "\t", $SNP{$snp}{chi2} . "\n";
  if($SNP{$snp}{chi2} > $SNP_chi2_cut){
    $GWS_SNP{$snp} = 1;
  }
}

print "Printing report file\n";
open (OUT, ">report.$model.txt");
print OUT join("\t", qw(GeneID GeneName Chr Start End Dist2Gene NearestGene nSNP Tests snp.chi2)); #Pritam
#print OUT join("\t", qw(GeneID GeneName Chr Start End Dist2Gene NearestGene nSNP Tests));

if(scalar(keys %minSNP) > 0){
  print OUT "\t" .  join("\t", qw(minSNP.chi2 minSNP.Pval));
}
if(scalar(keys %minSNPP) > 0){
  print OUT "\t" . join("\t", qw(minSNP.P.chi2 minSNP.P.Pval));
}
if(scalar(keys %GWiS) > 0){
  #print OUT "\t" . join("\t", qw(GWiS.k GWiS.Pval)); Pritam
  print OUT "\t" . join("\t", qw(GWiS.k GWiS.BIC GWiS.Pval));
}
if(scalar(keys %BimBam) > 0){
  print OUT "\t" . join("\t", qw(BimBam.sum BimBam.Pval));
}
if(scalar(keys %Vegas) > 0){
  print OUT "\t" . join("\t", qw(VEGAS.sum VEGAS.Pval));
}
if(scalar(keys %Gates) > 0){
  print OUT "\t" . join("\t", qw(Gates Gates.Pval));
}
print OUT "\n";

foreach my $g (sort{$GENE{$a}{chr} <=> $GENE{$b}{chr} || $GENE{$a}{start} <=> $GENE{$b}{start}} keys %GWS_genes){

  print OUT join("\t", $g, $GENE{$g}{name}, $GENE{$g}{chr}, $GENE{$g}{start}, $GENE{$g}{end}, 0, "-", $GENE{$g}{n},$GENE{$g}{v}, "-"); #Pritam
  #print OUT join("\t", $g, $GENE{$g}{name}, $GENE{$g}{chr}, $GENE{$g}{start}, $GENE{$g}{end}, 0, "-", $GENE{$g}{n},$GENE{$g}{v}); 

if(scalar(keys %minSNP) > 0){
  print OUT "\t" . join("\t",$minSNP{$g}{chi2},$minSNP{$g}{pval});
}
if(scalar(keys %minSNPP) > 0){
  print OUT "\t" . join("\t",$minSNPP{$g}{chi2}, $minSNPP{$g}{pval});
}
if(scalar(keys %GWiS) > 0){
  #print OUT "\t" .join("\t",$GWiS{$g}{k}, $GWiS{$g}{pval}); Pritam
  print OUT "\t" .join("\t",$GWiS{$g}{k}, $GWiS{$g}{bic}, $GWiS{$g}{pval});
}
if(scalar(keys %BimBam) > 0){
  print OUT "\t" . join("\t",$BimBam{$g}{BF_sum}, $BimBam{$g}{pval});
}
if(scalar(keys %Vegas) > 0){
  print OUT "\t" . join("\t",$Vegas{$g}{VEGAS_sum}, $Vegas{$g}{pval});
}
if(scalar(keys %Gates) > 0){
  print OUT "\t" . join("\t",$Gates{$g}{Gates}, $Gates{$g}{pval});
}
print OUT  "\n";
}

foreach my $snp (sort{$SNP{$a}{chr} <=> $SNP{$b}{chr} || $SNP{$a}{pos} <=> $SNP{$b}{pos}} keys %GWS_SNP){
  my ($gene, $dist, $dist_mid)=SNPdist2Gene($SNP{$snp}{chr}, $SNP{$snp}{pos},\%GENE);
  if($dist ==0){
    next;
  }
  print OUT join("\t", $snp, $snp, $SNP{$snp}{chr}, $SNP{$snp}{pos}, $SNP{$snp}{pos}, $dist ,$GWiS{$gene}{name}, 1,1,$SNP{$snp}{chi2});
  
  if(scalar(keys %minSNP) > 0){
    print OUT "\t" . join("\t","-", "-");
  }
  if(scalar(keys %minSNPP) > 0){
    print OUT "\t" . join("\t","-", "-");
  }
  if(scalar(keys %GWiS) > 0){
    print OUT "\t" .join("\t","-", "-","-");
  }
  if(scalar(keys %BimBam) > 0){
    print OUT "\t" . join("\t","-", "-");
  }
  if(scalar(keys %Vegas) > 0){
    print OUT "\t" . join("\t","-", "-");
  }
  if(scalar(keys %Gates) > 0){
    print OUT "\t" . join("\t","-", "-");
  }
  print OUT  "\n";
}

close(OUT);
